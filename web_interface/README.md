# Interfaz web del laboratorio ANFIS

Interfaz para configurar, ejecutar y comparar experimentos del modelo neurodifuso
sin escribir JSON a mano ni seguir el progreso en la terminal.

Envuelve los scripts del proyecto **sin modificarlos**: los importa y reorquesta el
barrido, así que los resultados son comparables con los de
`python rpipeline.py`. De hecho, una configuración creada aquí se puede descargar y
ejecutar con el script tal cual.

> Para el detalle de cómo está construido —arquitectura, patrones, API completa,
> decisiones y verificación— ver **[SRS.md](SRS.md)**.

---

## Instalación

Solo faltan las dependencias del servidor web; el resto ya está en el entorno `caic`.

```bash
conda activate caic
```

```bash
python -m pip install -r web_interface/requirements.txt
```

Instala `fastapi`, `uvicorn[standard]`, `pydantic`, `python-multipart` y `jinja2`.
No hace falta Node ni ningún paso de compilación: Chart.js viene incluido en
`estaticos/js/vendor/`.

---

## Arrancar

Desde la raíz del proyecto:

```bash
python -m web_interface.main
```

Abre <http://127.0.0.1:8000>.

Opciones:

```bash
python -m web_interface.main --puerto 8080
```

| Opción | Efecto |
|---|---|
| `--puerto N` | Puerto de escucha (por defecto 8000) |
| `--hilos N` | Hilos de PyTorch (por defecto, núcleos disponibles menos uno) |
| `--exponer` | Escucha en toda la red **y desactiva la edición de políticas** |

También se puede lanzar con uvicorn directamente, incluso desde otro directorio:

```bash
python -m uvicorn web_interface.presentacion.app:app --host 127.0.0.1 --port 8000
```

Si usas `--reload` para desarrollar, excluye `data/` o el recargador matará los
trabajos en curso al escribirse los resultados:

```bash
python -m uvicorn web_interface.presentacion.app:app --reload --reload-dir web_interface --reload-exclude 'web_interface/data/*'
```

---

## Recorrido rápido

### 1 · Elegir el dataset

En **Constructor**, selecciona un archivo del desplegable. La interfaz lo lee y
propone el esquema: tipo de problema, número de entradas y salidas, columna objetivo
y, si las clases son texto, el mapa de etiquetas a números. Con `iris.data` deduce
clasificación, 4 entradas, 3 salidas y las tres especies. Revisa la propuesta y
corrige lo que haga falta.

Los pares de `.mat` (como `engineInputs` / `engineTargets`) se emparejan solos.

### 2 · Añadir optimizadores

Cada optimizador trae su propio formulario, generado leyendo la firma real de su
constructor: si mañana cambias de versión de PyTorch, el formulario se actualiza
solo. Cada campo lleva una casilla **«por defecto»** marcada de inicio; solo se
envían los valores que desmarques, de modo que el diccionario de hiperparámetros
sea el mínimo.

Tres optimizadores aparecen marcados como incompatibles con este modelo (`LBFGS`,
`SparseAdam` y `Muon`); se pueden usar confirmando el aviso.

### 3 · Ajustar el entrenamiento

Rango de reglas, corridas, épocas, función de pérdida, particiones y políticas de
paro. Abajo verás el número total de entrenamientos y el **reparto efectivo** de los
datos.

> Un detalle heredado del script: `train_size` no interviene en la partición. Con
> `test_size = 0.2` y `val_size = 0.2` el reparto real es 80 % / 16 % / 4 %. La web
> reproduce ese comportamiento para que los números coincidan con el CLI, y te
> muestra las proporciones reales.

### 4 · Validar y ejecutar

**Validar** comprueba la configuración contra los registros y las firmas reales de
los constructores antes de gastar CPU. **Ejecutar ahora** encola el barrido y te
lleva al monitor.

Desde aquí también puedes **descargar el JSON** para usarlo con
`python rpipeline.py`, o **guardar la configuración** para reutilizarla.

### 5 · Seguir el entrenamiento

El monitor muestra el progreso global, la corrida actual, la curva de pérdida época
a época (con escala logarítmica, que es la que deja ver de verdad la convergencia),
la tabla de corridas terminadas y la salida del núcleo ya limpia de códigos ANSI.

- **Cancelar** corta al terminar la época en curso, en menos de un segundo, y
  **conserva los resultados parciales**.
- Puedes recargar la página o cerrarla: el trabajo sigue y al volver se reconstruye
  la vista completa.
- Se ejecuta un entrenamiento a la vez; los demás esperan en la cola.

### 6 · Comparar resultados

**Historial** reúne los experimentos lanzados desde la web y los que ya estaban en
`resultados/`. Cada uno abre un tablero con la tabla pivote optimizador × reglas, la
gráfica de la métrica que elijas y el CSV descargable. En los legados se muestran
además los PNG que generó el script.

---

## Añadir cosas nuevas

### Datasets

En **Datasets**, arrastra un archivo o selecciónalo. Se admiten `.csv`, `.data`,
`.txt`, `.xls`, `.xlsx` y `.mat`, hasta 200 MB. Quedan en
`web_interface/data/datasets/` y aparecen en el constructor. Los de `data_sets/` son
de solo lectura.

### Optimizadores

Deja un archivo `.py` en `web_interface/plugins/` con una clase que derive de
`torch.optim.Optimizer` y pulsa **Recargar plugins** en la página de Optimizadores.
El contrato completo está en [plugins/README.md](plugins/README.md); hay un ejemplo
funcionando en `plugins/ejemplo_optimizador.py`.

### Políticas de paro

Una política decide en cada época si detener el entrenamiento. En **Políticas**
escribes el nombre —del que sale el nombre de la clase— e implementas `apply` en el
editor, que ya viene prellenado con el esqueleto:

```python
class PoliticaPromedioTolerancia(PoliticaDeParo):
    def __init__(self) -> None:
        self.nombre = "Politica Promedio Tolerancia"

    def apply(self, loss_value: float, optimizador: Any) -> bool:
        return loss_value < 1e-9      # True detiene la corrida
```

**Validar y probar** comprueba la sintaxis y el contrato, ejecuta la política con una
secuencia de pérdidas y te enseña qué contestó a cada una:

```
100 → sigue · 10 → sigue · 1 → sigue · 0.1 → sigue · 1e-9 → detiene
```

que es la forma más rápida de ver si la condición está al revés. Nada se guarda sin
pasar esa prueba, y cada escritura respalda el archivo anterior.

Las políticas se guardan en `politicas/politicas_usuario.py`, que puedes editar con
tu editor y recargar desde la página. Una vez guardadas aparecen en el constructor
para elegirlas por experimento.

Dentro de `apply` tienes disponibles `np`, `torch` y `Any`. La pérdida llega ya
convertida a `float`, aunque el optimizador sea de PyTorch.

---

## Dónde queda cada cosa

| Ruta | Contenido |
|---|---|
| `data/resultados/<id>/` | Resultados de cada barrido: 5 archivos, incluidos `resultados.csv` y `resultados.json` con el mismo formato que el CLI |
| `data/configs/` | Configuraciones guardadas, en formato del CLI |
| `data/fis/<id>/` | Sistemas de inferencia difusa iniciales |
| `data/datasets/` | Datasets subidos |
| `data/cache/` | Caché del historial y respaldos del archivo de políticas |
| `politicas/politicas_usuario.py` | Tus políticas de paro |
| `plugins/` | Tus optimizadores |

Todo lo que genera la interfaz queda dentro de `web_interface/`. El directorio
`resultados/` del proyecto **solo se lee**, nunca se escribe.

> El JSON que descargas para el CLI sí apunta a `resultados/<dataset>/`, porque es la
> convención del script. Al ejecutarlo, `rpipeline.py` **sobrescribirá** el
> `resultados.csv` y el `resultados.json` que haya en esa carpeta. Cambia
> `resultados_path` en el archivo si quieres conservarlos.

---

## Sobre seguridad

El servidor escucha solo en `localhost` salvo que uses `--exponer`.

Los archivos de `plugins/` y `politicas/` **se ejecutan con los permisos de este
proceso**. Por eso el código de los plugins no se puede subir por la web —lo
depositas tú con tu editor y el endpoint solo re-escanea el directorio— y la edición
de políticas queda desactivada cuando el servidor no es local.

Para desactivar los plugins por completo, pon `PERMITIR_PLUGINS = False` en
`configuracion.py`.

---

## Solución de problemas

| Síntoma | Causa y solución |
|---|---|
| «MPS no compatible» al elegir dispositivo | El modelo usa `float64` y MPS no lo soporta. Usa CPU. |
| Un cambio en el CSS o el JS no se ve | No debería ocurrir: los estáticos van versionados por fecha. Si pasa, recarga con Ctrl+Shift+R. |
| Un plugin no aparece | Debe derivar de `torch.optim.Optimizer`, estar **definido** en su archivo (no solo importado) y no llamarse `nombre = "LM"`. Los errores se muestran en la página. |
| Una política no aparece | Pulsa **Recargar del archivo**. Si el archivo tiene un error de sintaxis, la página lo indica: ninguna política se carga hasta arreglarlo. |
| El trabajo desapareció tras reiniciar | La cola vive en memoria. Los resultados persistidos siguen en `data/resultados/`. |
| El puerto está ocupado | `python -m web_interface.main --puerto 8080` |

---

## Comprobar que todo está bien

```bash
curl -s localhost:8000/api/salud
```

Devuelve la raíz detectada, las versiones de Python y PyTorch y los dispositivos
disponibles. La documentación interactiva de las 55 rutas de la API está en
<http://127.0.0.1:8000/docs>.

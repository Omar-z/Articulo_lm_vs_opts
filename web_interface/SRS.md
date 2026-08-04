# Especificación del sistema — Interfaz web del laboratorio ANFIS

**Versión 0.1.0** · Documento de especificación y diseño

---

## Índice

1. [Introducción](#1-introducción)
2. [Descripción general](#2-descripción-general)
3. [Arquitectura](#3-arquitectura)
4. [Requisitos funcionales](#4-requisitos-funcionales)
5. [Diseño detallado y patrones](#5-diseño-detallado-y-patrones)
6. [Interfaces](#6-interfaces)
7. [Requisitos no funcionales](#7-requisitos-no-funcionales)
8. [Decisiones de diseño](#8-decisiones-de-diseño)
9. [Riesgos y limitaciones](#9-riesgos-y-limitaciones)
10. [Verificación](#10-verificación)
11. [Trabajo pendiente](#11-trabajo-pendiente)

---

## 1. Introducción

### 1.1 Propósito

Sustituir el flujo de trabajo por línea de comandos del laboratorio ANFIS por una
interfaz web, sin modificar los scripts que hacen el cálculo.

Antes, cada experimento exigía escribir a mano un archivo JSON con el esquema del
dataset, las particiones, el rango de reglas y los hiperparámetros exactos de cada
optimizador de PyTorch, lanzarlo con `python rpipeline.py iris_tests.json cuda
graficar lote`, seguir una barra de progreso ANSI en la terminal y abrir a mano los
CSV y PNG resultantes.

### 1.2 Alcance

El sistema cubre:

- construcción visual de experimentos, con inferencia automática del esquema del dataset;
- ejecución con progreso época a época y cancelación;
- comparación de resultados, incluidos los que ya existían en `resultados/`;
- alta de datasets nuevos;
- alta de optimizadores nuevos derivados de `torch.optim.Optimizer`;
- alta de políticas de paro escritas desde la propia web;
- barrido de learning rate (equivalente a `mejores_parametros.py`).

Queda **fuera**: modificar el núcleo de cálculo, entrenar en remoto, autenticación
multiusuario y despliegue en red pública.

### 1.3 Definiciones

| Término | Significado |
|---|---|
| **ANFIS** | Sistema de inferencia difusa adaptativa; aquí, Sugeno de orden 0 con reglas diagonales |
| **Núcleo** | Los scripts originales: `V2_Anfis.py`, `Anfis_utils.py`, `funciones_auxiliares.py` |
| **Regla** | Cada regla difusa; el barrido compara modelos con distinto número de reglas |
| **Corrida** | Una ejecución completa de entrenamiento para una terna (regla, optimizador, repetición) |
| **Barrido** | El recorrido completo de reglas × optimizadores × corridas |
| **Política de paro** | Estrategia que decide en cada época si detener el entrenamiento |
| **Trabajo** | Un barrido encolado, en ejecución o terminado |

### 1.4 Referencias

- Núcleo: `V2_Anfis.py`, `Anfis_utils.py`, `funciones_auxiliares.py`
- Especificación del barrido: `rpipeline.py` (se lee, no se importa — ver §8.2)
- Esquema del JSON de experimento: `README.md` de la raíz del proyecto

---

## 2. Descripción general

### 2.1 Restricción fundamental

**Todo el código nuevo vive en `web_interface/`.** El núcleo no se modifica: se
importa. Esta restricción condicionó el diseño entero y es el origen del mecanismo
descrito en §5.1.

### 2.2 Entorno verificado

| Componente | Versión |
|---|---|
| Python | 3.12.9 (entorno conda `caic`) |
| PyTorch | 2.12.1 — **no 2.6**; expone 15 optimizadores |
| FastAPI / uvicorn / pydantic | 0.140 / 0.51 / 2.13 |
| pandas / numpy / scipy / scikit-learn | 2.3 / 2.3 / 1.15 / 1.7 |
| fuzzylab | 0.13 |

### 2.3 Volumen del sistema

11.118 líneas repartidas en 51 módulos Python, 12 plantillas HTML, 8 archivos
JavaScript y 1 hoja de estilos. Sin Node, sin paso de compilación y sin
dependencias remotas en tiempo de ejecución.

---

## 3. Arquitectura

### 3.1 Capas

Cuatro capas con dependencias unidireccionales:

```
presentación  →  aplicación  →  dominio
     ↓               ↓
        infraestructura  →  dominio
                 ↓
              núcleo (importado, nunca modificado)
```

- **dominio** — modelos, estados, eventos y políticas. No importa FastAPI. Se puede
  probar sin levantar el servidor.
- **aplicación** — orquestación del barrido. No conoce HTTP ni el disco.
- **infraestructura** — único punto de contacto con el disco, `inspect` y el núcleo.
- **presentación** — FastAPI, plantillas Jinja2 y WebSockets.

### 3.2 Árbol de archivos

```
web_interface/
├── main.py                        Entrada: MPLBACKEND=Agg, hilos de torch, uvicorn
├── configuracion.py               Rutas, límites e interruptores
├── nucleo.py                      Única superficie de importación del núcleo
├── requirements.txt
│
├── dominio/
│   ├── modelos.py                 EventoProgreso, DescriptorOptimizador,
│   │                              ParticionDatos, ResultadoCorrida, ResumenExperimento
│   ├── estados.py                 Patrón State: 6 estados y sus transiciones
│   ├── eventos.py                 Patrón Observer: SujetoDeProgreso, HistorialDeEventos
│   ├── politicas_web.py           Patrón Strategy: observador, cancelación, adaptador
│   └── trabajo.py                 Trabajo: estado + bandera + emisor + historial
│
├── aplicacion/
│   ├── fachada_anfis.py           Facade sobre el núcleo
│   ├── barridos.py                Template Method: experimento / hiperparámetros
│   ├── comandos.py                Command: unidades encolables
│   ├── gestor_trabajos.py         Cola FIFO + hilo trabajador
│   ├── ensambladores.py           Adapter DTO ⇄ dataclasses ⇄ JSON del CLI
│   ├── servicio_datasets.py       Vista previa, estadísticas, inferencia de esquema
│   ├── servicio_optimizadores.py  Catálogo y recarga de plugins
│   └── servicio_politicas.py      Catálogo de políticas y construcción de prototipos
│
├── infraestructura/
│   ├── rutas.py                   Resolución respecto a la raíz, saneado, anti-traversal
│   ├── registros.py               Registry de optimizadores y funciones de pérdida
│   ├── introspeccion.py           Descripción de optimizadores por firma
│   ├── cargadores.py              Strategy + Factory por formato de archivo
│   ├── editor_fis.py              Creación del FIS inicial con ruta absoluta
│   ├── salida.py                  Multiplexor de stdout y bitácora sin ANSI
│   ├── dispositivos.py            CPU/CUDA/MPS y su compatibilidad
│   ├── repositorio_datasets.py    Catálogo de datasets y alta de archivos
│   ├── repositorio_experimentos.py Configuraciones guardadas
│   ├── repositorio_resultados.py  Repository + Composite sobre dos formatos
│   ├── repositorio_plugins.py     Carga de optimizadores del usuario
│   └── repositorio_politicas.py   Lectura, validación y escritura de políticas
│
├── presentacion/
│   ├── app.py                     create_app(): ciclo de vida y routers
│   ├── esquemas.py                DTO pydantic
│   ├── plantillas.py              Jinja2, filtros y versionado de estáticos
│   ├── puente_ws.py               Hilo → event loop
│   └── rutas/                     paginas + 6 routers de API + WebSockets
│
├── plantillas/                    12 plantillas Jinja2
├── estaticos/                     CSS, JS y Chart.js vendorizado
├── plugins/                       Optimizadores del usuario
├── politicas/                     Políticas de paro del usuario
└── data/                          Runtime (gitignored): configs, fis, resultados, caché
```

---

## 4. Requisitos funcionales

### RF-1 · Constructor de experimentos (`/constructor`)

| Id | Requisito |
|---|---|
| RF-1.1 | Seleccionar un dataset del catálogo y prellenar el esquema automáticamente |
| RF-1.2 | Mostrar una vista previa de las primeras filas |
| RF-1.3 | Añadir y quitar optimizadores, con formulario de hiperparámetros generado dinámicamente |
| RF-1.4 | Elegir las políticas de paro que se aplicarán |
| RF-1.5 | Calcular en vivo el número total de entrenamientos |
| RF-1.6 | Mostrar las proporciones **efectivas** de la partición (ver §9.3) |
| RF-1.7 | Validar la configuración sin ejecutarla |
| RF-1.8 | Exportar el JSON que acepta `rpipeline.py` |
| RF-1.9 | Guardar la configuración para reutilizarla |
| RF-1.10 | Confirmar con un aviso efímero cada optimizador añadido o quitado |

### RF-2 · Ejecución y monitor (`/trabajos`, `/monitor/{id}`)

| Id | Requisito |
|---|---|
| RF-2.1 | Encolar un barrido y responder de inmediato, sin bloquear el servidor |
| RF-2.2 | Publicar el progreso época a época |
| RF-2.3 | Dibujar la curva de pérdida en vivo, con escala logarítmica o lineal |
| RF-2.4 | Cancelar un trabajo en curso conservando los resultados parciales |
| RF-2.5 | Reconstruir la vista completa tras recargar la página |
| RF-2.6 | Recurrir a sondeo si el WebSocket cae, y reconectar con retroceso exponencial |
| RF-2.7 | Mostrar la salida por consola del núcleo sin códigos ANSI |
| RF-2.8 | Indicar el avance de la carga cuando hay muchas curvas |
| RF-2.9 | Ejecutar un solo entrenamiento a la vez; el resto espera en cola |

### RF-3 · Resultados (`/historial`)

| Id | Requisito |
|---|---|
| RF-3.1 | Listar en un mismo historial los experimentos de la web y los de `resultados/` |
| RF-3.2 | Mostrar la tabla pivote optimizador × reglas |
| RF-3.3 | Graficar cualquier métrica frente al número de reglas |
| RF-3.4 | Descargar el CSV |
| RF-3.5 | Mostrar los PNG que dejó el script en los experimentos legados |
| RF-3.6 | Persistir cada barrido en cinco archivos, incluso si se canceló |

### RF-4 · Datasets (`/datasets`)

| Id | Requisito |
|---|---|
| RF-4.1 | Listar los de `data_sets/` (solo lectura) y los subidos |
| RF-4.1.1 | De `data_sets/` cargar **solo los archivos preprocesados**, reconocidos por el sufijo `_pre` o `-pre` antes de la extensión. Los subidos por la web no pasan por el filtro |
| RF-4.2 | Vista previa con separador y encabezado ajustables |
| RF-4.3 | Estadísticas por columna: tipo, rango, media, nulos, distintos |
| RF-4.4 | Inferir el esquema: tipo de problema, entradas, salidas, columna objetivo y mapa de clases |
| RF-4.5 | Emparejar automáticamente los pares de `.mat` (entradas/objetivos) |
| RF-4.6 | Subir archivos nuevos con lista blanca de extensiones |

### RF-5 · Optimizadores (`/optimizadores`)

| Id | Requisito |
|---|---|
| RF-5.1 | Descubrir automáticamente los de `torch.optim` |
| RF-5.2 | Deducir sus hiperparámetros de la firma del constructor |
| RF-5.3 | Integrar el Levenberg–Marquardt propio, de firma distinta |
| RF-5.4 | Cargar optimizadores del usuario desde `plugins/` |
| RF-5.5 | Marcar los incompatibles con el motivo concreto |
| RF-5.6 | Validar los hiperparámetros contra la firma real antes de ejecutar |

### RF-6 · Políticas de paro (`/politicas`)

| Id | Requisito |
|---|---|
| RF-6.1 | Listar las cuatro del núcleo y las del usuario |
| RF-6.2 | Generar el esqueleto de la clase a partir del nombre que escribe el usuario |
| RF-6.3 | Validar sintaxis y contrato antes de guardar |
| RF-6.4 | Ejecutar la política con pérdidas de prueba y mostrar su respuesta |
| RF-6.5 | Guardar en un archivo Python legible y editable a mano |
| RF-6.6 | Recargar el archivo bajo demanda |
| RF-6.7 | Editar y borrar las del usuario; impedirlo en las del núcleo |
| RF-6.8 | Respaldar el archivo anterior en cada escritura |

### RF-7 · Barrido de hiperparámetros (`/hiperparametros`)

Barrido logarítmico de learning rate por optimizador y número de reglas,
equivalente a `mejores_parametros.py`. Fuera de la navegación principal, accesible
por URL.

---

## 5. Diseño detallado y patrones

### 5.1 El mecanismo que hace viable todo el sistema

El núcleo no expone ningún punto de extensión: `train_nfs` y `train_nfs_batch` no
aceptan callbacks. Pero **sí construyen un `CompositorDePoliticas` con la lista
`early_stop` y lo consultan en cada época** (`V2_Anfis.py:464` y `:554`):

```python
if(politicas_de_paro.apply(loss, optimizer)):
```

Inyectando en esa lista estrategias propias que cumplan el mismo `Protocol` que las
del núcleo (`funciones_auxiliares.py:290`) se obtiene progreso época a época y
cancelación cooperativa **sin modificar una sola línea de los scripts**.

Tres estrategias en `dominio/politicas_web.py`:

| Clase | Devuelve | Función |
|---|---|---|
| `PoliticaObservador` | siempre `False` | Publica un evento de progreso por época |
| `PoliticaCancelacion` | `True` al activarse la bandera | Corta cuando el usuario pulsa Cancelar |
| `PoliticaNormalizada` | lo que diga la envuelta | Adapta el tipo de la pérdida (ver §9.1) |

**El orden de la lista importa** y es obligatorio:

```
[observador, cancelacion, *deepcopy(politicas_seleccionadas)]
```

El observador va primero porque devuelve siempre `False` y así ve todas las épocas:
`CompositorDePoliticas` corta en cuanto una política devuelve `True`. La cancelación
va segunda para que se le atribuya el motivo del paro.

Solo se copian con `deepcopy` las políticas de paro reales —tienen estado mutable y
cada corrida necesita las suyas—; el observador y la cancelación se pasan **por
referencia**, porque duplicar el `threading.Event` dejaría sorda la cancelación.

### 5.2 Catálogo de patrones

| Patrón | Clases | Problema concreto que resuelve |
|---|---|---|
| **Strategy** | `PoliticaObservador`, `PoliticaCancelacion`, `PoliticaNormalizada` | El núcleo no tiene callbacks; es el único enganche por época |
| **Facade** | `FachadaANFIS` | Lo pide el propio código en `V2_Anfis.py:895`. Una corrida son ~40 líneas de pegamento entre seis módulos, duplicadas entre `rpipeline.main` y `mejores_parametros` |
| **Observer** | `SujetoDeProgreso`, `HistorialDeEventos` | Tres consumidores con ritmos distintos: historial completo, WebSockets diezmados y escritor de resultados |
| **Template Method** | `BarridoBase` → `BarridoDeExperimento`, `BarridoDeHiperparametros` | Los dos barridos comparten el 80 % del flujo; hoy los dos scripts lo duplican entero |
| **Registry** | `RegistroDeOptimizadores`, `RegistroDeFuncionesPerdida` | `OPTIMIZADORES` y `FN_LOSS` son diccionarios literales duplicados en dos archivos, y no admiten altas en caliente |
| **Repository** | `RepositorioDeResultados`, `...Experimentos`, `...Datasets`, `...Plugins`, `...Politicas` | Aísla el acceso a disco y permite unificar formatos distintos |
| **Composite** | `RepositorioDeResultados` con `FuenteWeb` + `FuenteRepoLegado` | El historial lista dos formatos de almacenamiento sin que los endpoints ramifiquen |
| **Adapter** | `EnsambladorDeConfig` | Tres representaciones de la misma configuración: DTO validado, dataclasses del núcleo y JSON del CLI |
| **State** | `EstadoTrabajo` + 6 estados | Cada endpoint decide si su acción es legal; sin él serían cadenas de `if` en tres archivos |
| **Command** | `ComandoDeExperimento`, `ComandoDeHiperparametros` | La cola necesita objetos autocontenidos; el gestor no debe conocer la firma de cada barrido |
| **Factory Method** | `FabricaDeCargadores` | Sustituye la cascada `if path[-3:] == ...` y la amplía |
| **Prototype** | `deepcopy` de las políticas por corrida | Son stateful; cada corrida necesita instancias limpias |

**Descartados**: Singleton (`Depends` + `app.state` basta; el núcleo ya demuestra el
daño de los globales de módulo), Abstract Factory (no hay familias de productos),
Decorator sobre políticas (`CompositorDePoliticas` ya compone), Chain of
Responsibility (el compositor *es* una cadena), Memento/Visitor (no hay deshacer).

### 5.3 Estados de un trabajo

```
Pendiente ──arrancar──▶ Ejecutando ──terminar──▶ Terminado
    │                       │
    │                       ├──cancelar──▶ Cancelando ──terminar──▶ Cancelado
    │                       └──fallar────▶ Fallido
    └──cancelar──▶ Cancelado
```

Cada estado declara `puede_cancelar()`, `puede_eliminar()` y `es_terminal()`. Una
transición no contemplada lanza `TransicionInvalida`, que la API traduce a 409.

### 5.4 Introspección de optimizadores

`IntrospectorDeOptimizadores` descubre las subclases de `torch.optim.Optimizer` y
lee `inspect.signature(cls.__init__)` para generar el formulario:

| Evidencia en la firma | Control | Ejemplo real |
|---|---|---|
| valor por defecto `bool` | casilla | `nesterov=False`, `amsgrad=False` |
| anotación `bool \| None` | selector auto/sí/no | `foreach=None`, `fused=None` |
| valor por defecto `tuple` | *n* campos numéricos | `betas=(0.9, 0.999)`, `Adafactor.eps=(None, 0.001)` |
| anotación con `int` y no `float` | entero | `max_iter=20`, `history_size=100` |
| anotación con `float`/`Tensor` | decimal | `lr`, `eps`, `weight_decay` |
| primer parámetro (`params`/`model`) | **se omite** | lo inyecta `RLANFISBuilder` |

El LM propio coexiste sin caso especial: `RLANFISBuilder.Build()` (`V2_Anfis.py:222`)
discrimina con `issubclass(cls, Optimizador)`. Verificado que
`LevenberMaquardtOpt` cumple esa condición y `torch.optim.Adam` no, así que un
plugin derivado de `torch.optim.Optimizer` cae solo en la rama correcta.

### 5.5 Editor de políticas

Flujo de alta:

```
nombre legible ──▶ nombre_a_clase() ──▶ plantilla_de_politica()
                                              │
                    el usuario implementa apply()
                                              │
                              validar() ─ ast: sintaxis, una sola clase,
                                          nombre coincidente, existe apply
                                              │
                              probar()  ─ ejecuta en módulo aislado,
                                          instancia sin argumentos,
                                          llama apply() con 7 pérdidas
                                              │
                              guardar() ─ respalda, inserta o reemplaza
                                          el bloque, revalida el archivo entero
```

Los bloques de clase se localizan con `ast` y no con expresiones regulares, para que
un decorador, un docstring que contenga la palabra `class` o una clase anidada no
despisten al editor ni al borrado.

### 5.6 Persistencia de resultados

Cada barrido terminado produce cinco archivos en `data/resultados/<id_trabajo>/`:

| Archivo | Contenido |
|---|---|
| `resultados.json` | Los 17 campos por celda, con las series completas — mismo formato que el CLI |
| `resultados.csv` | Tabla pivote `optimizador,metrica,regla_N` — mismo formato que el CLI |
| `resumen.json` | Solo los promedios; es lo que lee el historial |
| `config.json` | La configuración exacta, en formato del CLI |
| `info_experimentos.txt` | Descripción en prosa del protocolo |

---

## 6. Interfaces

### 6.1 Páginas

| Ruta | Contenido |
|---|---|
| `/` | Portada con accesos y estado del servidor |
| `/constructor` | Constructor visual de experimentos |
| `/trabajos` | Cola e historial de trabajos de la sesión |
| `/monitor/{id}` | Monitor en vivo |
| `/historial` | Experimentos terminados, propios y legados |
| `/historial/{origen}/{id}` | Tablero comparativo |
| `/datasets` · `/datasets/{id}` | Explorador y detalle |
| `/optimizadores` | Catálogo y plugins |
| `/politicas` | Catálogo y editor de políticas |
| `/hiperparametros` | Barrido de learning rate (fuera de la navegación) |

### 6.2 API JSON

**Trabajos**

| Método | Ruta | Función |
|---|---|---|
| POST | `/api/trabajos` | Encola un barrido. Responde 202 con `id_trabajo` |
| POST | `/api/trabajos/validar` | Valida una configuración sin ejecutarla |
| GET | `/api/trabajos` | Lista los trabajos |
| GET | `/api/trabajos/{id}` | Estado y progreso |
| POST | `/api/trabajos/{id}/cancelar` | Solicita el corte |
| DELETE | `/api/trabajos/{id}` | Elimina un trabajo terminado |
| GET | `/api/trabajos/{id}/eventos` | Eventos posteriores a una secuencia |
| GET | `/api/trabajos/{id}/curvas` | Varias curvas por lotes |
| GET | `/api/trabajos/{id}/curva` | Una curva concreta |
| GET | `/api/trabajos/{id}/bitacora` | Salida del núcleo sin ANSI |

**Políticas**

| Método | Ruta | Función |
|---|---|---|
| GET | `/api/politicas` | Catálogo completo |
| GET | `/api/politicas/plantilla` | Esqueleto a partir del nombre |
| POST | `/api/politicas/validar` | Valida y prueba sin escribir |
| POST | `/api/politicas` | Guarda o reemplaza |
| DELETE | `/api/politicas/{clase}` | Borra una del usuario |
| POST | `/api/politicas/recargar` | Relee el archivo |
| GET | `/api/politicas/archivo` | Contenido íntegro |

**Optimizadores, datasets, experimentos y resultados**

| Método | Ruta | Función |
|---|---|---|
| GET | `/api/optimizadores` · `/{nombre}` | Catálogo y descriptor |
| POST | `/api/optimizadores/validar` | Valida hiperparámetros contra la firma |
| POST | `/api/optimizadores/recargar-plugins` | Re-escanea `plugins/` |
| GET | `/api/datasets` | Catálogo |
| GET | `/api/datasets/{id}/vista-previa` · `/estadisticas` · `/esquema-sugerido` | Exploración |
| POST · DELETE | `/api/datasets` · `/{id}` | Alta y baja |
| GET/POST/PUT/DELETE | `/api/experimentos` … | Configuraciones guardadas |
| POST | `/api/experimentos/exportar` · `/importar` | Conversión con el formato del CLI |
| GET | `/api/resultados` · `/{origen}/{id}` · `/comparativa` · `/csv` | Historial |
| GET | `/api/salud` · `/api/dispositivos` · `/api/funciones-perdida` | Servicio |

En total 55 rutas. La documentación interactiva está en `/docs`.

### 6.3 WebSockets

| Ruta | Al conectar | Después |
|---|---|---|
| `/ws/trabajos/{id}` | `snapshot` con estado, inventario de curvas, corridas y bitácora | Eventos `epoca`, `corrida_inicio`, `corrida_fin`, `regla_fin`, `estado`, `bitacora`, `fin`, `error`, `latido` |
| `/ws/cola` | Lista de trabajos | Lista actualizada en cada cambio |

El cliente puede enviar `{"accion": "replay", "desde": n}`, `{"accion": "snapshot"}`
o `{"accion": "ping"}`.

---

## 7. Requisitos no funcionales

### 7.1 Concurrencia

**Un único hilo trabajador con cola FIFO.** No multiprocessing.

Justificación: las operaciones caras del núcleo (`matmul`, `linalg.solve`, `jacfwd`
en la jacobiana del LM, el `backward` de PyTorch) liberan el GIL, así que el event
loop sigue atendiendo peticiones mientras se entrena. Con hilos, además, la bandera
de cancelación y el emisor de eventos son memoria compartida directa; con procesos
cada evento de época exigiría serialización y en macOS el arranque `spawn`
reimportaría torch y fuzzylab en cada trabajo.

Se ejecuta **un entrenamiento a la vez** a propósito: compiten por la misma CPU y
los tiempos deben ser comparables entre optimizadores.

Contrapartida aceptada: no hay forma de matar un trabajo de golpe. La cancelación es
cooperativa con granularidad de época.

### 7.2 Rendimiento (medido)

| Operación | Medida |
|---|---|
| `POST /api/trabajos` | 1,5 ms |
| `GET /api/trabajos` **durante** un entrenamiento | < 1 ms |
| Cancelación: de la petición al estado `cancelado` | 0,04 s |
| Detalle de `wine` (cuyo `resultados.json` pesa 75 MB) | 1,4 ms |
| Listado del historial completo | 7 ms |
| Snapshot del monitor, 120 series | 50 KB |
| Lote de 25 curvas | 63 KB en 0,9 ms |
| Dibujo de 240 series × 500 puntos en Chart.js | 47 ms |

### 7.3 Aislamiento de la salida por consola

`train_nfs` lanza un hilo demonio (`mostrar_barra_progreso`, `V2_Anfis.py:736`) que
escribe secuencias ANSI en `sys.stdout` cada 100 ms. Sin tratamiento, eso deja
ilegibles los registros de uvicorn.

`contextlib.redirect_stdout` capturaría la barra, pero reemplaza `sys.stdout` de
forma **global** y también se tragaría los registros del servidor. La solución es un
multiplexor instalado una sola vez que enruta **por hilo**: el hilo principal (donde
vive el event loop, y por tanto uvicorn) escribe en la consola real; cualquier otro
hilo escribe en la bitácora del trabajo activo.

Se enruta por «hilo principal frente al resto» y no con `threading.local` porque el
hilo de la barra lo crea `train_nfs`, no esta aplicación, y no heredaría el valor.

Efecto secundario aprovechado: la línea `"Se detuvo por: …"` que imprime
`CompositorDePoliticas` queda capturada y se adjunta como motivo de paro.

### 7.4 Independencia del directorio de trabajo

`configuracion.RAIZ_PROYECTO` se calcula desde la ubicación del archivo y se añade a
`sys.path`. Las rutas relativas del JSON se resuelven con `resolver_relativa`, nunca
con `Path.resolve()` a secas. **Nunca se usa `os.chdir`**: es estado global del
proceso y corrompería las peticiones concurrentes.

`CrearFISInicial` escribe el archivo con el nombre que se le pasa tal cual
(`Anfis_utils.py:39`), así que se le entrega siempre un prefijo absoluto bajo
`data/fis/<id_trabajo>/`. Esta es la razón de que la raíz del repositorio acumulara
nueve archivos `*_inicial.fis__init.fis`.

### 7.5 Seguridad

El sistema es una herramienta local de investigación, no un servicio público. Las
mitigaciones son proporcionadas a ese contexto:

| Riesgo | Mitigación |
|---|---|
| Ejecución de código (plugins) | No se suben por la web; el usuario deposita el archivo con su editor y el endpoint solo re-escanea |
| Ejecución de código (políticas) | **Solo con el servidor en `127.0.0.1`**; con `--exponer` la escritura devuelve 403 |
| Código malformado | Validación con `ast` y ejecución de prueba antes de escribir |
| Pérdida de trabajo | Respaldo del archivo anterior en cada escritura de políticas |
| Path traversal | Los datasets se direccionan por identificador opaco; todo acceso resuelve con `Path.resolve()` y verifica `is_relative_to` |
| Subidas maliciosas | Lista blanca `{.csv,.data,.txt,.xls,.xlsx,.mat}`, tope de 200 MB, nombre saneado; esos archivos nunca se importan |
| Fallo de un plugin | Cada archivo se carga aislado; uno roto devuelve un error legible y no tumba el servidor |

No se intenta aislamiento real (subproceso restringido, lista blanca de AST): daría
una falsa sensación de seguridad y estorbaría a código legítimo que use numpy o torch.

### 7.6 Compatibilidad con el flujo por línea de comandos

Una configuración creada en la web se descarga como JSON y se ejecuta con
`python rpipeline.py archivo.json cpu` sin traducción. El CSV que produce la web
tiene el mismo formato pivote que el del script. **Verificado ejecutándolo.**

---

## 8. Decisiones de diseño

### 8.1 Fidelidad frente a corrección

Donde el núcleo hace algo discutible pero determinista, la web **reproduce el
comportamiento** en vez de corregirlo, para que los números sean comparables con los
del CLI. Los casos concretos están en §9.3 y §9.4, y la interfaz avisa de ellos.

### 8.2 No se importa `rpipeline.py`

Técnicamente funcionaría: su bloque `__main__` está guardado y `seaborn` está
instalado. Aun así se decidió no importarlo:

1. `main()` depende de cinco variables globales de módulo mutadas desde `__main__`,
   borra la pantalla con secuencias ANSI y lanza su propio hilo de barra.
2. `cargar_datos` tiene dos defectos que hay que evitar: para un `.mat` suelto
   incluye las claves internas `__header__`, `__version__` y `__globals__`; y para
   Excel pasa `sep` a `read_excel`, que no lo acepta.
3. Hacen falta cargadores más ricos igualmente (vista previa, estadísticas, inferencia).

Se importan solo `V2_Anfis`, `Anfis_utils` y `funciones_auxiliares`, y todo pasa por
`nucleo.py` para que quede en un solo lugar qué se usa del núcleo.

### 8.3 Carga incremental de curvas

El snapshot inicial **no incluye las curvas**. En un barrido completo (8 reglas × 6
optimizadores × 5 corridas) son 240 series de hasta 500 puntos: unos 2,7 MB que el
servidor tendría que serializar de golpe y el navegador esperar enteros antes de
mostrar nada.

En su lugar se envía el inventario (`claves_curvas`) y el cliente pide las curvas en
lotes de 25. Con 120 series, el snapshot bajó de 324 KB a 50 KB. Como el total se
conoce de antemano, la barra de progreso indica algo real en vez de ser decorativa.

Los eventos de época que lleguen durante la carga se encolan y se aplican al
terminar: cada lote reemplaza la serie entera y, sin esa cola, se perderían los
puntos recibidos en vivo.

### 8.4 Versionado de los archivos estáticos

Las plantillas referencian los estáticos con `{{ estatico('css/estilos.css') }}`,
que añade la fecha de modificación del archivo a la URL. Sin esto el navegador
conserva la copia cacheada del CSS o del JS aunque el archivo haya cambiado, y las
modificaciones parecen no surtir efecto. Es un fallo caro de diagnosticar: el código
está bien y el navegador sirve la versión vieja.

### 8.5 Sin singletons de módulo

Los componentes de larga vida se crean en el ciclo de vida y viven en `app.state`,
de donde los toman los endpoints. El núcleo ya demuestra el daño del antipatrón
contrario: `rpipeline.dispositivo`, `graficar`, `minilotes`, `mejores_parametros` y
`mp_path` son globales mutados desde `__main__`, y por eso `main()` es inutilizable
desde cualquier otro contexto.

---

## 9. Riesgos y limitaciones

### 9.1 Tipos de la pérdida entre el bucle y las políticas

Las políticas del núcleo se escribieron asumiendo que la pérdida llega como `float`,
que es lo que entrega el optimizador LM. Pero con los optimizadores de PyTorch
`train_nfs` pasa el `torch.Tensor` con grafo asociado (`V2_Anfis.py:440` y `:464`).
Esa diferencia provocó un fallo real en `PoliticaNanOrInf`:

```
np.isnan(tensor) → Tensor.__array__ → .numpy()
RuntimeError: Can't call numpy() on Tensor that requires grad
```

**Mitigación**: `PoliticaNormalizada` envuelve cada política del núcleo y convierte
la pérdida a `float` una sola vez antes de delegar. Así la web funciona con
cualquier política que se añada al núcleo, la escriba quien la escriba. (El núcleo
recibió después una corrección equivalente; el adaptador se mantiene por robustez y
porque además evita que `PoliticaFallos` conserve tensores entre épocas.)

### 9.2 Volumen de los resultados legados

Los `resultados.json` de `resultados/` pesan entre 23 y 75 MB, porque guardan las
series completas de todas las corridas. Cargar uno en un endpoint congelaría el
servidor.

**Mitigación**: el resumen sale siempre del `resultados.csv` (unos KB, ya contiene
todos los promedios) y se cachea por ruta y fecha de modificación. Las curvas se
sirven bajo demanda, submuestreadas a 500 puntos por decimación uniforme que
conserva el primer y el último valor.

### 9.3 `train_size` no interviene en la partición

`rpipeline.main` no usa ese campo: aparta `test_size` y subdivide el resto con
`val_size`. Con `0.6 / 0.2 / 0.2` el reparto real es **80 % / 16 % / 4 %**, no
60/20/20.

**Mitigación**: se replica el comportamiento exacto y el constructor muestra las
proporciones efectivas calculadas, con un aviso de que `train_size` es informativo.

### 9.4 Mezcla de tipos en clasificación

`OneHotEncode` devuelve float32 (`funciones_auxiliares.py:97`) mientras el modelo es
float64 (`V2_Anfis.py:33-35`). El pipeline convive con ello por promoción de tipos.

**Mitigación**: la fachada reproduce ese comportamiento por defecto y expone
`forzar_float64` como opción explícita.

### 9.5 MPS no es utilizable

El modelo crea todos sus parámetros en `torch.float64` y MPS no soporta ese tipo:
`.to("mps")` lanza `TypeError`. `/api/dispositivos` lo marca incompatible con el
motivo y el DTO lo rechaza.

### 9.6 Optimizadores incompatibles

Tres de los 15 de `torch.optim` no funcionan con este modelo y se marcan con su
motivo; se pueden seleccionar solo confirmando el aviso:

| Optimizador | Motivo |
|---|---|
| `LBFGS` | Necesita un `closure` en `step()`; `train_nfs` llama a `optimizer.step()` sin argumentos |
| `SparseAdam` | Solo admite gradientes dispersos; los del modelo son densos |
| `Muon` | Diseñado para matrices de peso 2D de redes profundas |

### 9.7 Colisión de nombres con el LM

`train_nfs:437` decide cómo invocar `step()` mirando el atributo `nombre` de la
instancia. Un optimizador de PyTorch que se llamara `"LM"` recibiría `step(X, y)` y
fallaría con `TypeError`. Se comprueba en dos sitios: al cargar el plugin (atributo
de clase) y tras construir el modelo (atributo de la instancia, porque
`LevenberMaquardtOpt` lo fija en `__init__`).

### 9.8 Los trabajos viven en memoria

Reiniciar el servidor pierde la cola y el historial de eventos de la sesión. Los
**resultados** persistidos no se pierden: están en `data/resultados/`.

### 9.9 `cantidad_reglas` del núcleo está roto

Su diccionario `estado` (`V2_Anfis.py:671-676`) no incluye la clave `"flair"`, que
`mostrar_barra_progreso:749` sí lee, de modo que el hilo de la barra muere con
`KeyError` en silencio. **No se usa**: el barrido de reglas es un bucle propio.

---

## 10. Verificación

Todo lo que sigue se comprobó ejecutándolo contra el entorno real.

### 10.1 El mecanismo de enganche

| Comprobación | Resultado |
|---|---|
| Observador con LM, Adam y SGD, 5 épocas | 5 eventos por corrida en los tres casos |
| Cancelación con la bandera activada | Cortó en **1 época** |
| Motivo de paro capturado de stdout | `Se detuvo por: Cancelación solicitada por el usuario` |

### 10.2 Ejecución y aislamiento

| Comprobación | Resultado |
|---|---|
| Latencia de `POST /api/trabajos` | 1,5 ms |
| Latencia de la API durante el entrenamiento | < 1 ms (el GIL se libera) |
| Cancelación de un barrido de 36 corridas | `cancelando` → `cancelado` en 0,04 s |
| Secuencias ANSI en la consola de uvicorn | **0** |
| Registros de acceso visibles | 26 |
| Bitácora del trabajo | Contiene la barra de progreso, ya sin escapes |
| `.fis` generados | Todos en `data/fis/<id>/`; `git status` limpio en la raíz |

### 10.3 Datos y catálogos

| Comprobación | Resultado |
|---|---|
| Cargadores contra los cinco formatos reales | CSV, `.data`, Excel, `.mat` suelto y par de `.mat`: todos correctos |
| Inferencia de esquema de `iris.data` | Clasificación, 4 entradas, 3 salidas, columna objetivo 4, mapa de las tres clases, 150 filas |
| Emparejado de `engineInputs.mat` + `engineTargets.mat` | 2 entradas y 2 salidas, con las dos rutas |
| Catálogo de optimizadores | 17: 15 de torch + LM + 1 plugin; 3 marcados incompatibles |
| `Adafactor.eps` | Tupla de aridad 2 con `[None, 0.001]` |
| `LM` | Solo `lambda_init`, `lambda_decr`, `lambda_incr`; sin `model` ni `device` |
| Path traversal (`/api/datasets/../../etc/passwd/...`) | 404 |

### 10.4 Extensibilidad

| Comprobación | Resultado |
|---|---|
| Plugin `SGDConDecaimiento` entrenando el dataset `engine` | Funcionó; R² 0,74–0,77 frente a 0,86–0,96 del LM |
| Política `PoliticaTopeDeEpocas(tope=37)` escrita desde la web, con máximo de 500 épocas | Las 4 corridas cortaron en **exactamente 37 épocas** |
| Motivo en la bitácora | `Se detuvo por: Politica Tope De Epocas` |
| Borrado de una política del núcleo | 403 con mensaje explicativo |
| Borrado de una política del usuario | 204; el archivo sigue siendo Python válido |
| Política inexistente al lanzar | 422 `Políticas desconocidas: NoExiste` |
| Errores de validación | Sintaxis, falta de `apply` y nombre discordante: los tres con mensaje accionable |

### 10.5 Resultados e interoperabilidad

| Comprobación | Resultado |
|---|---|
| Historial listando propios y legados | Los 4 legados en 7 ms |
| Detalle de `wine` (75 MB de JSON) | 1,4 ms, porque parsea el CSV |
| Archivos generados por un barrido | Los cinco, incluido el caso cancelado |
| JSON exportado ejecutado con `python rpipeline.py` | Completó y generó su CSV |
| Arranque desde otro directorio (`cd /tmp && uvicorn …`) | Detecta la raíz correctamente |

---

## 11. Trabajo pendiente

| Id | Descripción |
|---|---|
| TP-1 | **Métricas personalizadas**: mismo mecanismo que las políticas —archivo Python, editor con plantilla, validación con `ast` y prueba— aplicado a `RegistroDeFuncionesPerdida` y a las métricas que se calculan en cada época (`NOMBRES_METRICAS`) |
| TP-2 | Parámetros por política en el constructor: hoy las del usuario se instancian sin argumentos, así que todos sus parámetros necesitan valor por defecto |
| TP-3 | Persistir la cola de trabajos para que sobreviva a un reinicio |
| TP-4 | Matriz de confusión y gráficas de regresión en el detalle de resultados |
| TP-5 | Decidir el futuro del barrido de hiperparámetros, hoy fuera de la navegación |

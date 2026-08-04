# Optimizadores propios

Deja aquí un archivo `.py` y pulsa **Recargar** en `/optimizadores`. La clase
aparecerá en el catálogo con su formulario de hiperparámetros generado
automáticamente a partir de la firma de su constructor.

## Contrato

1. La clase debe derivar de **`torch.optim.Optimizer`**.
2. Debe estar **definida** en tu archivo, no solo importada. Si escribes
   `from torch.optim import Adam`, Adam no se vuelve a registrar.
3. Su `__init__` debe ser introspeccionable: parámetros con nombre y, salvo
   `params`, con valor por defecto. De esa firma sale el formulario.
4. **No definas `nombre = "LM"`.** Ese atributo es el discriminante con el que
   `train_nfs` decide cómo llamar a `step()` (`V2_Anfis.py:437`): con `"LM"` recibiría
   `step(X, y)` en lugar de `zero_grad()/backward()/step()`, y fallaría con `TypeError`.
5. Los archivos cuyo nombre empieza por `_` se ignoran.

## Cómo se traduce la firma al formulario

| Lo que escribes | Control que aparece |
|---|---|
| `lr: float = 0.01` | número decimal |
| `pasos: int = 5` | número entero |
| `nesterov: bool = False` | casilla |
| `foreach: bool \| None = None` | selector automático / sí / no |
| `betas=(0.9, 0.999)` | dos campos numéricos |

Los hiperparámetros llamados `lr`, `momentum`, `betas`, `eps`, `weight_decay` o
`alpha` se muestran expandidos; el resto queda plegado en «Avanzados».

## Un detalle del modelo

Los parámetros del ANFIS son `torch.float64` (`V2_Anfis.py:33-35`). Si tu
optimizador crea tensores de estado, créalos con el `dtype` del parámetro
(`torch.zeros_like(p)`) y no con `torch.float32`.

## Seguridad

Los archivos de esta carpeta **se ejecutan con los permisos del proceso del
servidor**. No se pueden subir por la web precisamente por eso: los depositas tú
con tu editor. Para desactivar el escaneo por completo, pon
`PERMITIR_PLUGINS = False` en `web_interface/configuracion.py`.

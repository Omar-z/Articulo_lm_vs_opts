# Como usar
El script [rpipeline.py](rpipeline.py) recibe como primer argumento un archivo `.json`
con toda la configuración del experimento (optimizadores, dataset, hiperparámetros, etc.):

```bash
python rpipeline.py iris_tests.json  # por defecto es cpu
python rpipeline.py iris_tests.json cuda # posibles valores cpu, cuda y mps
python rpipeline.py iris_tests.json cuda graficar # posibles graficar o nada
```

# Configuración Experimento

En este repo hay tres ejemplos listos para usar: [iris_tests.json](iris_tests.json),
[abalone_tests.json](abalone_tests.json) y [engine_tests.json](engine_tests.json).

El JSON tiene **dos** secciones obligatorias: `optimizadores` y `experimentacion`.

## `optimizadores`

Lista (array) de los optimizadores que se quieren comparar. Cada elemento es un objeto con:

| Campo    | Tipo   | Descripción |
|----------|--------|-------------|
| `name`   | string | Nombre del optimizador. Debe ser uno de los registrados en `OPTIMIZADORES`: `LM`, `SGD`, `Adam`, `AdamW`, `RMSprop`. |
| `params` | objeto | Hiperparámetros que se pasan **tal cual** al constructor del optimizador (`**params`). Las llaves deben coincidir con las que espera cada optimizador. |

Notas sobre `params`:

- `LM` es el optimizador propio (Levenberg–Marquardt) y usa `lambda_init`(equivalente al lr de los otros optimizadores), `lambda_decr`, `lambda_incr`. El campo `device` lo agrega el script automáticamente, no hay que ponerlo.
- `SGD`, `Adam`, `AdamW`, `RMSprop` son de PyTorch (`torch.optim`), así que sus `params` son los nombres de esa API: `lr`, `momentum`, `betas`, `eps`, `weight_decay`, `amsgrad`, `alpha`, `centered`, etc.
- Si pones un `name` que no está registrado, el script lanza un error.
- Posibles optimizadores a utilizar son $\rightarrow$ https://docs.pytorch.org/docs/2.12/optim.html#algorithms

```json
{
  "name": "Adam",
  "params": { "lr": 0.01, "betas": [0.9, 0.999], "eps": 1e-8 }
}
```

## `experimentacion`

Objeto único con la definición del dataset y los hiperparámetros de entrenamiento. Cada llave corresponde a un campo de `DataExperimento` en [rpipeline.py](rpipeline.py).

### Dataset

| Campo                 | Tipo                   | Descripción |
|-----------------------|------------------------|-------------|
| `dataset_path`        | string \| [string, string] | Ruta al archivo de datos. Soporta `.csv`, `.data` y `.mat`. Si las entradas y salidas están en archivos separados (caso `.mat` tipo MATLAB), se pasa una lista `["inputs.mat", "targets.mat"]`. |
| `dataset_header`      | int \| null \| objeto  | Para CSV/`.data` se pasa tal cual a `pandas.read_csv(header=...)` (`null` = sin encabezado). Para pares `.mat` es un objeto `{"in": "<clave_inputs>", "out": "<clave_targets>"}` con los nombres de las variables dentro del `.mat`. |
| `dataset_sep`      | str \| null  | Para CSV/`.data`/`.txt` se pasa tal cual a `pandas.read_csv(sep=...)` (`null` = separador `,`)|
| `dataset_target_col`  | int \| null            | Índice (0-based) de la columna objetivo. Si es `null`, las salidas se toman por posición usando `dataset_entradas` y `dataset_salidas`. |
| `dataset_map_col`     | objeto \| null         | Mapeo de etiquetas de texto a números para la columna objetivo (p. ej. `{"Iris-setosa": 0, ...}`). `null` si el target ya es numérico. Requiere que `dataset_target_col` no sea `null`. |
| `dataset_entradas`    | int                    | Número de características de entrada (columnas) del modelo. |
| `dataset_salidas`     | int                    | Número de salidas. En clasificación equivale al número de clases (se aplica one-hot). |

### Entrenamiento y resultados

| Campo             | Tipo   | Descripción |
|-------------------|--------|-------------|
| `resultados_path` | string | Carpeta donde se guardan `resultados.csv` e `info_experimentos.txt`. Se crea si no existe. Conviene terminar con `/`. |
| `corridas`        | int    | Número de veces que se entrena/evalúa cada combinación; los resultados se promedian. |
| `epocas`          | int    | Máximo de épocas por entrenamiento. |
| `tolerancia`      | float  | Umbral de la función de pérdida para detener antes (early stop), p. ej. `1e-12`. |
| `fallos`          | int    | `max_fails` de validación: número de épocas seguidas sin mejorar antes de detener. |
| `funcion_perdida` | string | Función de pérdida. Una de: `MSE`, `MAE`, `SSE`, `Entropia`. se puden usar las de pytorch o crear una función propia|
| `tipo`            | string | `"regresion"` o `"clasificacion"`. Define las métricas (R² vs exactitud/precisión) o si se aplica one-hot. |
| `reglas_inicial`  | int    | Número de reglas difusas con el que empieza el barrido. |
| `reglas_total`    | int    | Número de reglas con el que termina el barrido (inclusivo). Se evalúa de `reglas_inicial` a `reglas_total`. |
| `train_size`      | float  | Proporción de datos para entrenamiento (0–1). |
| `test_size`       | float  | Proporción para prueba (0–1). |
| `val_size`        | float  | Proporción para validación (0–1). |

### Ejemplo mínimo

```json
{
  "optimizadores": [
    { "name": "LM", "params": { "lambda_init": 0.01, "lambda_decr": 0.9, "lambda_incr": 10 }},
    {"name": "SGD", "params": { "lr": 0.01, "momentum": 0.9 }},
    {"name": "Adam","params": { "lr": 0.001, "betas":[0.9,0.999], "eps": 1e-8}}
  ],
  "experimentacion": {
    "dataset_path": "data_sets/iris/iris.data",
    "dataset_header": null,
    "dataset_sep":null,
    "dataset_target_col": 4,
    "dataset_map_col": { "Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2 },
    "dataset_entradas": 4,
    "dataset_salidas": 3,
    "resultados_path": "resultados/iris/",
    "corridas": 5,
    "epocas": 1000,
    "tolerancia": 1e-12,
    "fallos": 10,
    "funcion_perdida": "SSE",
    "tipo": "clasificacion",
    "reglas_inicial": 3,
    "reglas_total": 10,
    "train_size": 0.6,
    "test_size": 0.2,
    "val_size": 0.2
  }
}
```

## Datasets
En el folder de [data_sets/](data_sets) se encuentran los problemas que se quieren comparar. No todos los archivos estan limpios por lo que se te tiene que hacer un preprocesado y de preferencia que se quede el archivo en `.csv` para no batallar con el script al cargar los datos.

### Preprocesar

| Tipo | Preprocesamiento |
|------|------------------|
| fechas | las fechas será quitarlas o convertirlas a su equivalencia `Unix Timestamp`|
| clases | normalizarlas de `0 a N-1` para que el OneHotEncode no falle |
| nulos | de preferencia quitarlos o rellenarlos|
| mulitples archivos | juntarlos en uno solo archivo `.csv` y con el target en las ultimas columnas |



# Resultados
Se genera una tabla de los resultados en la dirección puesta en la variable `resultados_path` del archivo de configuración del experimento. La tabla contiene los siguientes campos

| campo | descripción |
|------|------------------|
| optimizador | el optimizador a probar|
| métrica | la meérica que se esta evaluando|
| $regla_x$ - $regla_y$ | el valor de la métrica utilizando `x` cantidad de reglas|

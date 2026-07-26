"""Única superficie de importación de los scripts del proyecto.

Todo lo que la interfaz web toma del núcleo pasa por aquí, para que quede en un solo
lugar qué se usa y con qué contrato. Los módulos importados son exactamente tres:

* `V2_Anfis`            — modelo, builder, optimizador LM y bucles de entrenamiento
* `Anfis_utils`         — creación, carga y guardado del FIS
* `funciones_auxiliares`— métricas, dataclasses de configuración y políticas de paro

**`rpipeline` no se importa a propósito.** Su `main()` depende de cinco variables
globales de módulo mutadas desde ``__main__``, escribe con rutas relativas al
directorio de trabajo, borra la pantalla con secuencias ANSI y lanza su propio hilo
de barra de progreso. Además su `cargar_datos` incluye las claves internas
(`__header__`, `__version__`, `__globals__`) al leer un `.mat` suelto y pasa `sep` a
`read_excel`, que no lo acepta. Se lee como especificación del barrido, pero la
interfaz reimplementa esa parte en `aplicacion/barridos.py`.

`configuracion` se importa primero porque fija ``MPLBACKEND`` y añade la raíz del
proyecto a ``sys.path``.
"""

from __future__ import annotations

import web_interface.configuracion as _configuracion  # noqa: F401  (efectos de import)

# --- V2_Anfis --------------------------------------------------------------------
from V2_Anfis import (  # noqa: E402
    ANFISND,
    LevenberMaquardtOpt,
    Optimizador,
    RLANFISBuilder,
    train_nfs,
    train_nfs_batch,
)

# --- Anfis_utils -----------------------------------------------------------------
from Anfis_utils import CargarFIS, CrearFISInicial, GuardarFIS  # noqa: E402

# --- funciones_auxiliares --------------------------------------------------------
from funciones_auxiliares import (  # noqa: E402
    CompositorDePoliticas,
    DataConfig,
    DataExperimento,
    DataOptimizador,
    OneHotEncode,
    PlotTraining,
    PoliticaDeParo,
    PoliticaFallos,
    PoliticaLambdaLM,
    PoliticaNanOrInf,
    PoliticaTolerancia,
    confusion_matrix,
)

__all__ = [
    "ANFISND",
    "CargarFIS",
    "CompositorDePoliticas",
    "CrearFISInicial",
    "DataConfig",
    "DataExperimento",
    "DataOptimizador",
    "GuardarFIS",
    "LevenberMaquardtOpt",
    "OneHotEncode",
    "Optimizador",
    "PlotTraining",
    "PoliticaDeParo",
    "PoliticaFallos",
    "PoliticaLambdaLM",
    "PoliticaNanOrInf",
    "PoliticaTolerancia",
    "RLANFISBuilder",
    "confusion_matrix",
    "train_nfs",
    "train_nfs_batch",
]

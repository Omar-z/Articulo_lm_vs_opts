"""Ajustes globales de la interfaz web.

Este módulo debe importarse **antes** que cualquier módulo del núcleo
(`V2_Anfis`, `Anfis_utils`, `funciones_auxiliares`), porque:

1. Fija ``MPLBACKEND=Agg``. `funciones_auxiliares` hace ``import matplotlib.pyplot``
   a nivel de módulo y en macOS podría resolver al backend ``MacOSX``, que aborta el
   proceso si se dibuja fuera del hilo principal.
2. Añade la raíz del proyecto a ``sys.path``, de modo que el núcleo se importa
   correctamente sin depender del directorio de trabajo desde el que se arranque.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#matplotlib sin interfaz gráfica: se fija antes de que nadie importe pyplot
os.environ.setdefault("MPLBACKEND", "Agg")

#raiz del repositorio (el padre de web_interface/)
RAIZ_PROYECTO: Path = Path(__file__).resolve().parents[1]
DIR_WEB: Path = Path(__file__).resolve().parent

#el nucleo vive en la raiz; se hace importable desde cualquier CWD
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

#--- Directorios de artefactos generados por la web -------------------------------
#Todo lo que produce la interfaz vive aqui dentro; nunca se escribe fuera.
DIR_DATOS: Path = DIR_WEB / "data"
DIR_CONFIGS: Path = DIR_DATOS / "configs"
DIR_FIS: Path = DIR_DATOS / "fis"
DIR_DATASETS_WEB: Path = DIR_DATOS / "datasets"
DIR_RESULTADOS_WEB: Path = DIR_DATOS / "resultados"
DIR_HIPERPARAMETROS: Path = DIR_DATOS / "hiperparametros"
DIR_CACHE: Path = DIR_DATOS / "cache"

#--- Directorios del repositorio que se leen (jamás se escriben) ------------------
DIR_DATASETS_REPO: Path = RAIZ_PROYECTO / "data_sets"
DIR_RESULTADOS_REPO: Path = RAIZ_PROYECTO / "resultados"

#--- Recursos de la propia aplicación ---------------------------------------------
DIR_PLANTILLAS: Path = DIR_WEB / "plantillas"
DIR_ESTATICOS: Path = DIR_WEB / "estaticos"
DIR_PLUGINS: Path = DIR_WEB / "plugins"
DIR_POLITICAS: Path = DIR_WEB / "politicas"
ARCHIVO_POLITICAS: Path = DIR_POLITICAS / "politicas_usuario.py"

#--- Servidor ---------------------------------------------------------------------
HOST: str = "127.0.0.1"
PUERTO: int = 8000

#Lo fija `main.py` al arrancar. Escribir políticas desde la web es ejecutar código
#en este proceso, así que solo se permite mientras el servidor sea local.
EXPUESTO_EN_RED: bool = False

#--- Límites ----------------------------------------------------------------------
#Subida de datasets
EXTENSIONES_DATASET: frozenset[str] = frozenset(
    {".csv", ".data", ".txt", ".xls", ".xlsx", ".mat"}
)
MAX_BYTES_DATASET: int = 200 * 1024 * 1024  # 200 MB

#Del directorio `data_sets/` solo se cargan los archivos ya preprocesados, es
#decir, los que terminan con uno de estos sufijos antes de la extensión
#(`abalone_pre.csv`, `wine-pre.csv`). El resto son datos crudos —con fechas,
#nulos o clases sin normalizar— que el modelo no puede consumir tal cual, y
#llenaban el catálogo de ruido: `bank_marketing` aporta seis archivos por sí solo.
#Los datasets que se suben desde la web no pasan por este filtro.
SUFIJOS_PREPROCESADO: tuple[str, ...] = ("_pre", "-pre")

#Eventos de progreso
MAX_EVENTOS_HISTORIAL: int = 20_000     # buffer circular por trabajo
MAX_LINEAS_BITACORA: int = 2_000        # líneas de stdout capturadas por trabajo
MAX_PUNTOS_CURVA: int = 500             # submuestreo antes de mandar al navegador
EVENTOS_POR_CORRIDA: int = 200          # objetivo de eventos de época por corrida
MAX_COLA_SUSCRIPTOR: int = 500          # eventos en vuelo por WebSocket
MAX_TRABAJOS_EN_MEMORIA: int = 50       # historial de trabajos de la sesión

#--- Plugins ----------------------------------------------------------------------
#Los archivos de plugins/ se ejecutan con los permisos de este proceso.
#Poner en False para no escanear el directorio.
PERMITIR_PLUGINS: bool = True

#--- Dispositivos -----------------------------------------------------------------
#El modelo ANFIS usa float64 en todos sus parámetros (V2_Anfis.py:33-35) y MPS no
#soporta ese dtype, así que se marca como incompatible en la API.
MOTIVO_MPS_INCOMPATIBLE: str = (
    "El modelo ANFIS usa float64 y MPS no soporta ese tipo de dato"
)


def asegurar_directorios() -> None:
    """Crea los directorios de artefactos si no existen."""
    for directorio in (
        DIR_DATOS,
        DIR_CONFIGS,
        DIR_FIS,
        DIR_DATASETS_WEB,
        DIR_RESULTADOS_WEB,
        DIR_HIPERPARAMETROS,
        DIR_CACHE,
    ):
        directorio.mkdir(parents=True, exist_ok=True)

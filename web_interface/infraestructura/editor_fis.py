"""Creación y saneado del sistema de inferencia difusa inicial.

`CrearFISInicial` escribe el archivo con ``fz.writeFIS(fis, nombre + "__init.fis")``
(`Anfis_utils.py:39`), es decir, abre el nombre **tal cual**. Si se le pasa un
nombre suelto, el `.fis` aterriza en el directorio de trabajo — por eso la raíz del
repositorio acumula hoy nueve archivos `*_inicial.fis__init.fis`.

Esta clase le entrega siempre un prefijo absoluto bajo `data/fis/<id_trabajo>/`, de
modo que la interfaz no ensucia el repositorio.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from web_interface.configuracion import DIR_FIS
from web_interface.infraestructura.rutas import sanear_nombre
from web_interface.nucleo import CrearFISInicial

#En algunos datasets el rango de una entrada es degenerado y fuzzylab escribe
#`[nan nan]` como parámetros de la función de pertenencia. Equivale al arreglo de
#`rpipeline.fis_nan_values_edit:601`.
_PARAMS_NAN = re.compile(r"\[\s*nan\s+nan\s*\]", re.IGNORECASE)


class EditorDeFIS:
    """Genera el FIS inicial de una combinación (dataset, número de reglas)."""

    def __init__(self, directorio_base: Path | None = None) -> None:
        self.directorio_base = directorio_base or DIR_FIS

    def crear_inicial(
        self,
        id_trabajo: str,
        nombre_dataset: str,
        regla: int,
        entradas: np.ndarray,
        salidas: np.ndarray,
    ) -> str:
        """Crea el `.fis` inicial y devuelve su ruta absoluta.

        @param entradas: matriz de características usada para calcular los rangos
        @param salidas: matriz de objetivos (ya en one-hot si es clasificación)
        @return: ruta absoluta del archivo escrito
        """
        destino = self.directorio_base / id_trabajo
        destino.mkdir(parents=True, exist_ok=True)

        base = sanear_nombre(nombre_dataset, "dataset")
        prefijo = str(destino / f"{base}_r{regla}")

        _, ruta = CrearFISInicial(
            prefijo,
            pd.DataFrame(entradas),
            pd.DataFrame(np.asarray(salidas).squeeze()),
            regla,
        )
        self.sanear_nan(ruta)
        return ruta

    @staticmethod
    def sanear_nan(ruta: str | Path) -> bool:
        """Reemplaza los parámetros `[nan nan]` por `[0.0 1.0]`.

        Un NaN en el FIS inicial se propaga a los parámetros del modelo y la corrida
        entera sale NaN. Devuelve True si hubo algo que corregir.
        """
        archivo = Path(ruta)
        try:
            contenido = archivo.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False

        corregido = _PARAMS_NAN.sub("[0.0 1.0]", contenido)
        if corregido != contenido:
            archivo.write_text(corregido, encoding="utf-8")
            return True
        return False

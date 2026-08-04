"""Servicio de datasets: vista previa, estadísticas e inferencia de esquema.

Lo que más fricción produce al configurar un experimento a mano es acertar con
`dataset_target_col`, `dataset_map_col`, `dataset_entradas` y `dataset_salidas`.
Aquí se deducen del propio archivo y se ofrecen como propuesta, que el usuario
puede corregir en el formulario.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from web_interface.infraestructura.cargadores import FabricaDeCargadores
from web_interface.infraestructura.repositorio_datasets import (
    Dataset,
    RepositorioDeDatasets,
)

#Con más valores distintos que esto, una columna entera se considera continua.
_MAX_CLASES = 20
#Filas que se leen para inferir el esquema y calcular estadísticas.
_FILAS_MUESTRA = 2000


class ServicioDeDatasets:
    """Operaciones de lectura sobre los datasets del catálogo."""

    def __init__(
        self,
        repositorio: RepositorioDeDatasets,
        fabrica: FabricaDeCargadores,
    ) -> None:
        self.repositorio = repositorio
        self.fabrica = fabrica

    # -- vista previa --------------------------------------------------------

    def opciones_inferidas(self, dataset: Dataset) -> dict[str, Any]:
        cargador = self.fabrica.para(dataset.ruta)
        return cargador.inferir_opciones(dataset.ruta)

    def vista_previa(
        self, id_dataset: str, opciones: dict[str, Any] | None, filas: int = 20
    ) -> dict[str, Any]:
        """Primeras filas del archivo, con las opciones de lectura aplicadas."""
        dataset = self.repositorio.obtener(id_dataset)
        cargador = self.fabrica.para(dataset.ruta)
        efectivas = {**self.opciones_inferidas(dataset), **(opciones or {})}

        marco = cargador.vista_previa(dataset.ruta, efectivas, filas)
        return {
            "id": dataset.id,
            "nombre": dataset.nombre,
            "ruta_relativa": dataset.a_dict()["ruta_relativa"],
            "columnas": [str(c) for c in marco.columns],
            "filas": _filas_serializables(marco),
            "n_columnas": int(marco.shape[1]),
            "n_filas_muestra": int(marco.shape[0]),
            "n_filas_total": self._contar_filas(dataset, efectivas),
            "opciones_inferidas": _serializable(efectivas),
        }

    def estadisticas(
        self, id_dataset: str, opciones: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Resumen por columna: tipo, rango, nulos y valores distintos."""
        marco = self._muestra(id_dataset, opciones)
        columnas: list[dict[str, Any]] = []

        for nombre in marco.columns:
            serie = marco[nombre]
            ficha: dict[str, Any] = {
                "nombre": str(nombre),
                "nulos": int(serie.isna().sum()),
                "n_unicos": int(serie.nunique(dropna=True)),
            }
            if pd.api.types.is_numeric_dtype(serie):
                limpia = serie.dropna()
                ficha.update(
                    {
                        "tipo": "numerica",
                        "min": _finito(limpia.min()),
                        "max": _finito(limpia.max()),
                        "media": _finito(limpia.mean()),
                        "desviacion": _finito(limpia.std()),
                        "histograma": _histograma(limpia),
                    }
                )
            else:
                conteo = serie.value_counts().head(12)
                ficha.update(
                    {
                        "tipo": "categorica",
                        "valores": [
                            {"valor": str(v), "cuenta": int(c)}
                            for v, c in conteo.items()
                        ],
                    }
                )
            columnas.append(ficha)

        return {"columnas": columnas, "n_filas_muestra": int(marco.shape[0])}

    # -- inferencia de esquema -----------------------------------------------

    def esquema_sugerido(
        self, id_dataset: str, opciones: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Propone la configuración del dataset a partir de su contenido.

        Heurística, en el orden en que se aplica:

        1. Si la última columna es de texto, es la clase: clasificación, con el
           mapa de etiquetas a enteros que necesita `dataset_map_col`.
        2. Si es numérica entera y con pocos valores distintos, también es clase.
        3. En cualquier otro caso, regresión con una sola salida.
        """
        dataset = self.repositorio.obtener(id_dataset)
        marco = self._muestra(id_dataset, opciones)
        n_columnas = int(marco.shape[1])

        if n_columnas < 2:
            raise ValueError(
                f"'{dataset.nombre}' solo tiene {n_columnas} columna(s); "
                "se necesitan al menos dos"
            )

        indice_objetivo = n_columnas - 1
        objetivo = marco[marco.columns[indice_objetivo]]
        aviso = ""

        if not pd.api.types.is_numeric_dtype(objetivo):
            etiquetas = sorted(str(v) for v in objetivo.dropna().unique())
            esquema = {
                "tipo": "clasificacion",
                "dataset_target_col": indice_objetivo,
                "dataset_map_col": {etiqueta: i for i, etiqueta in enumerate(etiquetas)},
                "dataset_entradas": n_columnas - 1,
                "dataset_salidas": len(etiquetas),
            }
            aviso = (
                f"La última columna es de texto con {len(etiquetas)} valores "
                "distintos; se ha interpretado como la clase"
            )
        elif _parece_clase(objetivo):
            valores = sorted(int(v) for v in objetivo.dropna().unique())
            esquema = {
                "tipo": "clasificacion",
                "dataset_target_col": indice_objetivo,
                "dataset_map_col": None,
                "dataset_entradas": n_columnas - 1,
                "dataset_salidas": len(valores),
            }
            if valores and (min(valores) != 0 or max(valores) != len(valores) - 1):
                aviso = (
                    f"Las clases van de {min(valores)} a {max(valores)}; el "
                    "codificador one-hot espera valores consecutivos desde 0. "
                    "Conviene renumerarlas antes de entrenar"
                )
            else:
                aviso = (
                    f"La última columna toma {len(valores)} valores enteros; "
                    "se ha interpretado como la clase"
                )
        else:
            esquema = {
                "tipo": "regresion",
                "dataset_target_col": None,
                "dataset_map_col": None,
                "dataset_entradas": n_columnas - 1,
                "dataset_salidas": 1,
            }
            aviso = (
                "La última columna es continua; se ha interpretado como una "
                "regresión con una salida"
            )

        pareja = self.repositorio.emparejar_mat(id_dataset)
        rutas: Any = dataset.a_dict()["ruta_relativa"]
        if pareja is not None:
            #En un par de .mat, un archivo aporta las entradas y el otro los objetivos.
            archivo_entradas, archivo_salidas = (
                (dataset, pareja)
                if "in" in dataset.ruta.stem.lower()
                else (pareja, dataset)
            )
            rutas = [
                archivo_entradas.a_dict()["ruta_relativa"],
                archivo_salidas.a_dict()["ruta_relativa"],
            ]

            #Las dimensiones salen de cada archivo por separado: mirar solo uno
            #daría la mitad de las columnas y una salida inventada.
            n_entradas = self._columnas_de(archivo_entradas.ruta)
            n_salidas = self._columnas_de(archivo_salidas.ruta)
            esquema = {
                "tipo": "regresion",
                "dataset_target_col": None,
                "dataset_map_col": None,
                "dataset_entradas": n_entradas,
                "dataset_salidas": n_salidas,
            }
            aviso = (
                f"Se ha emparejado con '{pareja.nombre}': "
                f"{archivo_entradas.nombre} aporta {n_entradas} entrada(s) y "
                f"{archivo_salidas.nombre}, {n_salidas} salida(s)"
            )

        return {
            **esquema,
            "dataset_path": rutas,
            "n_columnas": n_columnas,
            "n_filas": self._contar_filas(dataset, opciones or {}),
            "aviso": aviso,
            **{
                clave: valor
                for clave, valor in self.opciones_inferidas(dataset).items()
                if clave in ("sep", "header")
            },
        }

    # -- internos ------------------------------------------------------------

    def _muestra(
        self, id_dataset: str, opciones: dict[str, Any] | None
    ) -> pd.DataFrame:
        dataset = self.repositorio.obtener(id_dataset)
        cargador = self.fabrica.para(dataset.ruta)
        efectivas = {**self.opciones_inferidas(dataset), **(opciones or {})}
        return cargador.vista_previa(dataset.ruta, efectivas, _FILAS_MUESTRA)

    def _columnas_de(self, ruta: Path) -> int:
        """Número de columnas de un archivo suelto, leyendo solo unas filas."""
        cargador = self.fabrica.para(ruta)
        return int(cargador.vista_previa(ruta, cargador.inferir_opciones(ruta), 5).shape[1])

    def _contar_filas(self, dataset: Dataset, opciones: dict[str, Any]) -> int | None:
        """Cuenta las filas sin cargar el archivo en memoria, cuando se puede."""
        if dataset.extension in (".csv", ".data", ".txt"):
            try:
                with open(dataset.ruta, encoding="utf-8", errors="replace") as archivo:
                    total = sum(1 for linea in archivo if linea.strip())
            except OSError:
                return None
            return total - 1 if opciones.get("header") == 0 else total
        try:
            cargador = self.fabrica.para(dataset.ruta)
            return int(cargador.cargar(dataset.ruta, opciones).shape[0])
        except Exception:  # noqa: BLE001 - el conteo es informativo
            return None


def _parece_clase(serie: pd.Series) -> bool:
    """Una columna numérica con pocos valores enteros distintos es una etiqueta."""
    limpia = serie.dropna()
    if limpia.empty:
        return False
    if not np.all(np.equal(np.mod(limpia.to_numpy(dtype=float), 1), 0)):
        return False
    return int(limpia.nunique()) <= _MAX_CLASES


def _histograma(serie: pd.Series, cubetas: int = 20) -> dict[str, list[float]]:
    try:
        cuentas, bordes = np.histogram(serie.to_numpy(dtype=float), bins=cubetas)
    except (ValueError, TypeError):
        return {"cuentas": [], "bordes": []}
    return {
        "cuentas": [int(c) for c in cuentas],
        "bordes": [_finito(b) or 0.0 for b in bordes],
    }


def _finito(valor: Any) -> float | None:
    """JSON no admite NaN ni infinitos."""
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    return numero if math.isfinite(numero) else None


def _filas_serializables(marco: pd.DataFrame) -> list[list[Any]]:
    filas: list[list[Any]] = []
    for registro in marco.itertuples(index=False, name=None):
        filas.append(
            [
                None
                if (isinstance(v, float) and not math.isfinite(v)) or pd.isna(v)
                else (float(v) if isinstance(v, (np.floating, float)) else
                      int(v) if isinstance(v, (np.integer,)) else str(v))
                for v in registro
            ]
        )
    return filas


def _serializable(datos: dict[str, Any]) -> dict[str, Any]:
    return {
        clave: (valor if isinstance(valor, (str, int, float, bool, dict, list)) or valor is None
                else str(valor))
        for clave, valor in datos.items()
    }

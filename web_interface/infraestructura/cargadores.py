"""Carga de datasets — patrón Estrategia + Método Fábrica.

Sustituye la cascada de `if path[-3:] == ...` de `rpipeline.cargar_datos` (`:96-116`)
por una estrategia por formato, y la amplía con lo que necesita el explorador de
datasets: vista previa sin cargar el archivo entero e inferencia de opciones.

Corrige además dos defectos del original que no se pueden arreglar en su sitio:

* un `.mat` suelto se convertía con ``pd.DataFrame(loadmat(path))``, incluyendo las
  claves internas ``__header__``, ``__version__`` y ``__globals__``;
* para Excel se llamaba ``read_excel(path, **data_config)`` con un ``sep`` dentro,
  que `read_excel` no acepta y produce `TypeError`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy.io import loadmat


class FormatoNoSoportado(ValueError):
    """No hay ningún cargador para la extensión pedida."""


class CargadorDeDatos(Protocol):
    """Contrato de una estrategia de carga."""

    extensiones: tuple[str, ...]

    def cargar(self, ruta: Any, opciones: dict[str, Any]) -> pd.DataFrame: ...
    def vista_previa(
        self, ruta: Any, opciones: dict[str, Any], filas: int = 20
    ) -> pd.DataFrame: ...
    def inferir_opciones(self, ruta: Any) -> dict[str, Any]: ...


class CargadorCSV:
    """Texto delimitado: `.csv`, `.data` y `.txt`."""

    extensiones = (".csv", ".data", ".txt")

    def cargar(self, ruta: Path, opciones: dict[str, Any]) -> pd.DataFrame:
        return pd.read_csv(ruta, **self._limpiar(opciones))

    def vista_previa(
        self, ruta: Path, opciones: dict[str, Any], filas: int = 20
    ) -> pd.DataFrame:
        return pd.read_csv(ruta, nrows=filas, **self._limpiar(opciones))

    def inferir_opciones(self, ruta: Path) -> dict[str, Any]:
        """Adivina separador y presencia de encabezado leyendo las primeras líneas."""
        muestra = ""
        try:
            with open(ruta, encoding="utf-8", errors="replace") as archivo:
                muestra = "".join(next(archivo, "") for _ in range(20))
        except OSError:
            return {"sep": ",", "header": None}

        separador = ","
        try:
            separador = csv.Sniffer().sniff(muestra, delimiters=",;\t| ").delimiter
        except csv.Error:
            #Sin pistas: se elige el candidato que aparezca de forma más consistente.
            lineas = [l for l in muestra.splitlines() if l.strip()][:5]
            if lineas:
                mejor, mejor_cuenta = ",", 0
                for candidato in (",", ";", "\t", " "):
                    cuentas = {l.count(candidato) for l in lineas}
                    if len(cuentas) == 1 and (c := cuentas.pop()) > mejor_cuenta:
                        mejor, mejor_cuenta = candidato, c
                separador = mejor

        return {"sep": separador, "header": 0 if self._parece_encabezado(muestra, separador) else None}

    @staticmethod
    def _parece_encabezado(muestra: str, separador: str) -> bool:
        """Hay encabezado si la primera fila no es numérica y la segunda sí."""
        lineas = [l for l in muestra.splitlines() if l.strip()]
        if len(lineas) < 2:
            return False

        def es_numerica(linea: str) -> bool:
            campos = [c.strip() for c in linea.split(separador) if c.strip()]
            if not campos:
                return False
            numericos = 0
            for campo in campos:
                try:
                    float(campo)
                    numericos += 1
                except ValueError:
                    pass
            return numericos >= max(1, len(campos) - 1)

        return not es_numerica(lineas[0]) and es_numerica(lineas[1])

    @staticmethod
    def _limpiar(opciones: dict[str, Any]) -> dict[str, Any]:
        """Deja pasar solo lo que entiende `read_csv`.

        `header` se conserva aunque valga None, porque ese es precisamente el valor
        que indica "el archivo no tiene fila de encabezado".
        """
        permitidas = {"sep", "header", "names", "skiprows", "na_values", "decimal"}
        return {
            clave: valor
            for clave, valor in opciones.items()
            if clave in permitidas and (valor is not None or clave == "header")
        }


class CargadorExcel:
    """Hojas de cálculo: `.xls` y `.xlsx`. Nunca recibe `sep`."""

    extensiones = (".xls", ".xlsx")

    def cargar(self, ruta: Path, opciones: dict[str, Any]) -> pd.DataFrame:
        return pd.read_excel(ruta, **self._limpiar(opciones))

    def vista_previa(
        self, ruta: Path, opciones: dict[str, Any], filas: int = 20
    ) -> pd.DataFrame:
        return pd.read_excel(ruta, nrows=filas, **self._limpiar(opciones))

    def inferir_opciones(self, ruta: Path) -> dict[str, Any]:
        return {"header": 0, "sheet_name": 0}

    @staticmethod
    def _limpiar(opciones: dict[str, Any]) -> dict[str, Any]:
        #`sep` es propio de read_csv; pasarlo a read_excel lanza TypeError.
        permitidas = {"header", "sheet_name", "skiprows", "names", "na_values"}
        return {k: v for k, v in opciones.items() if k in permitidas}


class CargadorMatSuelto:
    """Un único `.mat` de MATLAB.

    Descarta las claves internas de scipy (`__header__`, `__version__`,
    `__globals__`) y aplana los arreglos 2D en columnas.
    """

    extensiones = (".mat",)

    def cargar(self, ruta: Path, opciones: dict[str, Any]) -> pd.DataFrame:
        contenido = loadmat(ruta)
        matrices = {
            clave: valor
            for clave, valor in contenido.items()
            if not clave.startswith("__") and isinstance(valor, np.ndarray)
        }
        if not matrices:
            raise FormatoNoSoportado(f"El archivo '{ruta.name}' no contiene matrices")

        bloques: list[np.ndarray] = []
        nombres: list[str] = []
        for clave, matriz in matrices.items():
            datos = np.atleast_2d(matriz)
            #Formato MATLAB: características en filas, muestras en columnas.
            if datos.shape[0] < datos.shape[1]:
                datos = datos.T
            bloques.append(datos)
            nombres.extend(
                [clave] if datos.shape[1] == 1 else
                [f"{clave}_{i}" for i in range(datos.shape[1])]
            )

        filas = min(b.shape[0] for b in bloques)
        return pd.DataFrame(
            np.hstack([b[:filas] for b in bloques]), columns=nombres
        )

    def vista_previa(
        self, ruta: Path, opciones: dict[str, Any], filas: int = 20
    ) -> pd.DataFrame:
        return self.cargar(ruta, opciones).head(filas)

    def inferir_opciones(self, ruta: Path) -> dict[str, Any]:
        return {}


class CargadorMatPar:
    """Par de `.mat` separados en entradas y salidas, como el dataset `engine`.

    Réplica del caso de `rpipeline.cargar_datos:98-107`: las matrices vienen en
    formato MATLAB (características en filas) y se apilan verticalmente antes de
    transponer.
    """

    extensiones = (".mat",)

    def cargar(self, rutas: list[Path], opciones: dict[str, Any]) -> pd.DataFrame:
        clave_in, clave_out = self._claves(rutas, opciones)
        entradas = loadmat(rutas[0])[clave_in]
        salidas = loadmat(rutas[1])[clave_out]
        datos = np.vstack([entradas, salidas])
        return pd.DataFrame(datos.T)

    def vista_previa(
        self, rutas: list[Path], opciones: dict[str, Any], filas: int = 20
    ) -> pd.DataFrame:
        return self.cargar(rutas, opciones).head(filas)

    def inferir_opciones(self, rutas: list[Path]) -> dict[str, Any]:
        """Deduce el nombre de la variable dentro de cada `.mat`."""
        claves = []
        for ruta in rutas[:2]:
            contenido = loadmat(ruta)
            candidatas = [c for c in contenido if not c.startswith("__")]
            claves.append(candidatas[0] if candidatas else "")
        return {"header": {"in": claves[0], "out": claves[1] if len(claves) > 1 else ""}}

    @staticmethod
    def _claves(rutas: list[Path], opciones: dict[str, Any]) -> tuple[str, str]:
        encabezado = opciones.get("header")
        if isinstance(encabezado, dict) and "in" in encabezado and "out" in encabezado:
            return str(encabezado["in"]), str(encabezado["out"])

        #Sin claves explícitas: se toma la primera variable real de cada archivo.
        claves = []
        for ruta in rutas[:2]:
            contenido = loadmat(ruta)
            candidatas = [c for c in contenido if not c.startswith("__")]
            if not candidatas:
                raise FormatoNoSoportado(f"'{ruta.name}' no contiene ninguna variable")
            claves.append(candidatas[0])
        return claves[0], claves[1]


class FabricaDeCargadores:
    """Elige la estrategia de carga según la ruta (o el par de rutas)."""

    def __init__(self) -> None:
        self._csv = CargadorCSV()
        self._excel = CargadorExcel()
        self._mat = CargadorMatSuelto()
        self._mat_par = CargadorMatPar()

    def para(self, ruta: Path | list[Path]) -> CargadorDeDatos:
        if isinstance(ruta, (list, tuple)):
            if len(ruta) != 2:
                raise FormatoNoSoportado(
                    "Un dataset dividido debe tener exactamente dos archivos "
                    "(entradas y salidas)"
                )
            return self._mat_par

        extension = ruta.suffix.lower()
        for cargador in (self._csv, self._excel, self._mat):
            if extension in cargador.extensiones:
                return cargador
        raise FormatoNoSoportado(f"No hay cargador para archivos '{extension}'")

    def extensiones_soportadas(self) -> set[str]:
        return set(self._csv.extensiones) | set(self._excel.extensiones) | set(
            self._mat.extensiones
        )

    def cargar(self, ruta: Path | list[Path], opciones: dict[str, Any]) -> pd.DataFrame:
        """Atajo: elige la estrategia y carga."""
        return self.para(ruta).cargar(ruta, opciones)

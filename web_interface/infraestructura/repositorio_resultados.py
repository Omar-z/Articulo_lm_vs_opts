"""Persistencia y lectura de resultados — patrón Repositorio + Compuesto.

Hay que listar en un mismo historial dos formatos de almacenamiento distintos:

* el **legado** en `resultados/<dataset>/`, que produce `rpipeline.py` — un
  `resultados.csv` pivote, un `resultados.json` enorme y carpetas de PNG;
* el **propio** en `web_interface/data/resultados/<id_trabajo>/`, que añade un
  `resumen.json` pequeño y la configuración que lo generó.

Sin una interfaz común, cada endpoint tendría que ramificar por origen.

**Los `resultados.json` legados pesan entre 23 y 75 MB** porque guardan las series
completas de pérdida de todas las corridas. Cargar uno en un endpoint congelaría el
servidor, así que el resumen se obtiene siempre del CSV —unos pocos KB, y ya
contiene todos los promedios— y se cachea por ruta y fecha de modificación.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Protocol

from web_interface.configuracion import (
    DIR_CACHE,
    DIR_RESULTADOS_REPO,
    DIR_RESULTADOS_WEB,
)
from web_interface.dominio.modelos import ResumenExperimento
from web_interface.infraestructura.rutas import clave_de_cache, ruta_segura
from web_interface.nucleo import DataConfig, DataExperimento

#Orden de las filas del CSV, igual que `rpipeline.guardar_resultados:453-463`.
_FILAS_COMUNES = ("loss", "SSE", "MSE", "RMSE", "MAE", "epocas")
_FILAS_CLASIFICACION = ("exactitud", "presición")
_FILAS_REGRESION = ("R2",)

#Qué campo del acumulador alimenta cada fila del CSV.
_CAMPO_POR_FILA = {
    "loss": "prom_loss",
    "SSE": "prom_SSE",
    "MSE": "prom_MSE",
    "RMSE": "prom_RMSE",
    "MAE": "prom_MAE",
    "epocas": "prom_epocas",
    "exactitud": "prom_acc",
    "presición": "prom_prec",
    "R2": "prom_r2",
}


class ResultadoNoEncontrado(KeyError):
    """No existe el experimento pedido."""


class FuenteDeResultados(Protocol):
    """Contrato de un almacén de resultados."""

    origen: str

    def listar(self) -> list[ResumenExperimento]: ...
    def obtener(self, id_experimento: str) -> ResumenExperimento: ...
    def ruta_csv(self, id_experimento: str) -> Path: ...


# --------------------------------------------------------------------------------
# Fuente propia de la web
# --------------------------------------------------------------------------------

class FuenteWeb:
    """Resultados generados por la interfaz, en `data/resultados/<id_trabajo>/`."""

    origen = "web"

    def __init__(self, directorio: Path | None = None) -> None:
        self.directorio = directorio or DIR_RESULTADOS_WEB

    def listar(self) -> list[ResumenExperimento]:
        resumenes: list[ResumenExperimento] = []
        if not self.directorio.exists():
            return resumenes
        for carpeta in self.directorio.iterdir():
            if not carpeta.is_dir():
                continue
            try:
                resumenes.append(self._leer_resumen(carpeta))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return resumenes

    def obtener(self, id_experimento: str) -> ResumenExperimento:
        carpeta = ruta_segura(self.directorio, id_experimento)
        if not carpeta.is_dir():
            raise ResultadoNoEncontrado(f"No existe el resultado '{id_experimento}'")
        return self._leer_resumen(carpeta)

    def ruta_csv(self, id_experimento: str) -> Path:
        return ruta_segura(self.directorio, id_experimento, "resultados.csv")

    def ruta_json_completo(self, id_experimento: str) -> Path:
        return ruta_segura(self.directorio, id_experimento, "resultados.json")

    def _leer_resumen(self, carpeta: Path) -> ResumenExperimento:
        archivo = carpeta / "resumen.json"
        if not archivo.exists():
            raise ResultadoNoEncontrado(f"'{carpeta.name}' no tiene resumen.json")
        datos = json.loads(archivo.read_text(encoding="utf-8"))
        return ResumenExperimento(
            id=carpeta.name,
            origen=self.origen,
            nombre=datos.get("nombre", carpeta.name),
            dataset=datos.get("dataset", ""),
            tipo=datos.get("tipo", "regresion"),
            creado=datos.get("creado", carpeta.stat().st_mtime),
            optimizadores=datos.get("optimizadores", []),
            reglas=datos.get("reglas", []),
            metricas=datos.get("metricas", {}),
            cancelado=datos.get("cancelado", False),
            ruta=str(carpeta),
        )

    # -- escritura -----------------------------------------------------------

    def guardar(
        self,
        id_trabajo: str,
        config: DataConfig,
        resultado: dict[str, Any],
        config_cruda: dict[str, Any] | None = None,
    ) -> Path:
        """Escribe los cuatro archivos de un experimento terminado.

        Se guarda siempre, incluso si el barrido se canceló: un trabajo de horas no
        debe perder lo que ya calculó.
        """
        carpeta = ruta_segura(self.directorio, id_trabajo)
        carpeta.mkdir(parents=True, exist_ok=True)

        exp = config.experimentos
        estadisticas = resultado.get("estadisticas", {})

        #1. Volcado completo, con las series de todas las corridas.
        (carpeta / "resultados.json").write_text(
            json.dumps(estadisticas, indent=4, default=_json_seguro), encoding="utf-8"
        )

        #2. CSV pivote, mismo formato que el del CLI.
        self._escribir_csv(carpeta / "resultados.csv", exp, estadisticas)

        #3. Resumen ligero: es lo que lee el historial, para no abrir el JSON grande.
        metricas = self._extraer_promedios(exp, estadisticas)
        resumen = {
            "nombre": f"{_nombre_dataset(exp)} · reglas {exp.reglas_inicial}-{exp.reglas_total}",
            "dataset": str(exp.dataset_path),
            "tipo": exp.tipo,
            "creado": time.time(),
            "optimizadores": list(estadisticas.keys()),
            "reglas": list(range(exp.reglas_inicial, exp.reglas_total + 1)),
            "metricas": metricas,
            "cancelado": bool(resultado.get("cancelado")),
            "corridas": exp.corridas,
            "epocas": exp.epocas,
            "funcion_perdida": exp.funcion_perdida,
        }
        (carpeta / "resumen.json").write_text(
            json.dumps(resumen, indent=2, ensure_ascii=False, default=_json_seguro),
            encoding="utf-8",
        )

        #4. Configuración exacta, para poder repetir el experimento.
        if config_cruda is not None:
            (carpeta / "config.json").write_text(
                json.dumps(config_cruda, indent=2, ensure_ascii=False, default=_json_seguro),
                encoding="utf-8",
            )

        #5. Descripción en prosa, como `info_experimentos.txt` del CLI.
        self._escribir_info(carpeta / "info_experimentos.txt", exp, resumen)

        return carpeta

    @staticmethod
    def _escribir_csv(
        destino: Path, exp: DataExperimento, estadisticas: dict[str, Any]
    ) -> None:
        reglas = list(range(exp.reglas_inicial, exp.reglas_total + 1))
        filas = list(_FILAS_COMUNES) + list(
            _FILAS_CLASIFICACION if exp.tipo == "clasificacion" else _FILAS_REGRESION
        )

        with open(destino, "w", newline="", encoding="utf-8") as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(
                ["optimizador", "metrica"] + [f"regla_{r}" for r in reglas]
            )
            for optimizador, por_regla in estadisticas.items():
                if not por_regla:
                    continue
                for fila in filas:
                    campo = _CAMPO_POR_FILA[fila]
                    valores = [
                        _valor(por_regla.get(str(regla), {}).get(campo)) for regla in reglas
                    ]
                    escritor.writerow([optimizador, fila] + valores)

    @staticmethod
    def _extraer_promedios(
        exp: DataExperimento, estadisticas: dict[str, Any]
    ) -> dict[str, Any]:
        """Estructura {métrica: {optimizador: [valor por regla]}} para las gráficas."""
        reglas = [str(r) for r in range(exp.reglas_inicial, exp.reglas_total + 1)]
        nombres = list(_FILAS_COMUNES) + list(
            _FILAS_CLASIFICACION if exp.tipo == "clasificacion" else _FILAS_REGRESION
        )
        return {
            metrica: {
                optimizador: [
                    _valor(por_regla.get(regla, {}).get(_CAMPO_POR_FILA[metrica]))
                    for regla in reglas
                ]
                for optimizador, por_regla in estadisticas.items()
            }
            for metrica in nombres
        }

    @staticmethod
    def _escribir_info(
        destino: Path, exp: DataExperimento, resumen: dict[str, Any]
    ) -> None:
        proporcion_prueba = exp.test_size * (1 - exp.val_size)
        proporcion_val = exp.test_size * exp.val_size
        lineas = [
            f"Los experimentos para {exp.dataset_path} se realizaron de la forma:",
            f"\t- Se entrenó y probó {exp.corridas} vez/veces; se reporta el promedio",
            "\t- La partición usa la misma semilla para todos los optimizadores, "
            "de modo que la comparación sea justa",
            f"\t- Reparto efectivo: {1 - exp.test_size:.0%} entrenamiento / "
            f"{proporcion_prueba:.0%} prueba / {proporcion_val:.0%} validación",
            f"\t- Función de pérdida: {exp.funcion_perdida}; máximo {exp.epocas} épocas",
            f"\t- Optimizadores comparados: {', '.join(resumen['optimizadores'])}",
            f"\t- Reglas evaluadas: de {exp.reglas_inicial} a {exp.reglas_total}",
        ]
        if resumen.get("cancelado"):
            lineas.append("\t- ATENCIÓN: el barrido se canceló; los resultados son parciales")
        destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------
# Fuente legada del repositorio
# --------------------------------------------------------------------------------

class FuenteRepoLegado:
    """Resultados que ya existen en `resultados/`. **Solo lectura.**"""

    origen = "repo"

    def __init__(
        self, directorio: Path | None = None, directorio_cache: Path | None = None
    ) -> None:
        self.directorio = directorio or DIR_RESULTADOS_REPO
        self.directorio_cache = directorio_cache or DIR_CACHE

    def listar(self) -> list[ResumenExperimento]:
        resumenes: list[ResumenExperimento] = []
        if not self.directorio.exists():
            return resumenes
        for carpeta in sorted(self.directorio.iterdir()):
            if not carpeta.is_dir() or not (carpeta / "resultados.csv").exists():
                continue
            try:
                resumenes.append(self._resumen_de(carpeta))
            except (OSError, ValueError):
                continue
        return resumenes

    def obtener(self, id_experimento: str) -> ResumenExperimento:
        carpeta = ruta_segura(self.directorio, id_experimento)
        if not (carpeta / "resultados.csv").exists():
            raise ResultadoNoEncontrado(
                f"No existe el resultado legado '{id_experimento}'"
            )
        return self._resumen_de(carpeta)

    def ruta_csv(self, id_experimento: str) -> Path:
        return ruta_segura(self.directorio, id_experimento, "resultados.csv")

    def graficas(self, id_experimento: str) -> list[dict[str, str]]:
        """PNG que dejó el CLI, agrupados por carpeta."""
        carpeta = ruta_segura(self.directorio, id_experimento)
        imagenes: list[dict[str, str]] = []
        for subcarpeta in ("loss_promedio", "metricas_promedio", "loss", "metricas"):
            destino = carpeta / subcarpeta
            if not destino.is_dir():
                continue
            for imagen in sorted(destino.glob("*.png")):
                imagenes.append(
                    {
                        "grupo": subcarpeta,
                        "nombre": imagen.stem,
                        "url": f"/api/resultados/repo/{id_experimento}/grafica/{subcarpeta}/{imagen.name}",
                    }
                )
        return imagenes

    def ruta_grafica(self, id_experimento: str, grupo: str, nombre: str) -> Path:
        return ruta_segura(self.directorio, id_experimento, grupo, nombre)

    # -- internos ------------------------------------------------------------

    def _resumen_de(self, carpeta: Path) -> ResumenExperimento:
        csv_path = carpeta / "resultados.csv"
        cache = self.directorio_cache / f"{clave_de_cache(csv_path)}.json"

        if cache.exists():
            try:
                datos = json.loads(cache.read_text(encoding="utf-8"))
                return self._desde_dict(carpeta, datos)
            except (OSError, json.JSONDecodeError):
                pass

        datos = self._parsear_csv(csv_path)
        datos["info"] = self._leer_info(carpeta / "info_experimentos.txt")
        try:
            self.directorio_cache.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(datos), encoding="utf-8")
        except OSError:
            pass
        return self._desde_dict(carpeta, datos)

    def _desde_dict(self, carpeta: Path, datos: dict[str, Any]) -> ResumenExperimento:
        return ResumenExperimento(
            id=carpeta.name,
            origen=self.origen,
            nombre=f"{carpeta.name} (resultados del script)",
            dataset=datos.get("dataset", carpeta.name),
            tipo=datos.get("tipo", "regresion"),
            creado=(carpeta / "resultados.csv").stat().st_mtime,
            optimizadores=datos.get("optimizadores", []),
            reglas=datos.get("reglas", []),
            metricas=datos.get("metricas", {}),
            ruta=str(carpeta),
        )

    @staticmethod
    def _parsear_csv(ruta: Path) -> dict[str, Any]:
        """Lee el CSV pivote `optimizador,metrica,regla_N,...`."""
        with open(ruta, newline="", encoding="utf-8") as archivo:
            filas = list(csv.reader(archivo))

        if not filas:
            raise ValueError(f"'{ruta}' está vacío")

        cabecera = filas[0]
        reglas = [
            int(columna.split("_")[1])
            for columna in cabecera[2:]
            if columna.startswith("regla_")
        ]

        metricas: dict[str, dict[str, list[float | None]]] = {}
        optimizadores: list[str] = []
        for fila in filas[1:]:
            if len(fila) < 3:
                continue
            optimizador, metrica = fila[0], fila[1]
            if optimizador not in optimizadores:
                optimizadores.append(optimizador)
            metricas.setdefault(metrica, {})[optimizador] = [
                _a_float(v) for v in fila[2 : 2 + len(reglas)]
            ]

        tipo = "clasificacion" if "exactitud" in metricas else "regresion"
        return {
            "optimizadores": optimizadores,
            "reglas": reglas,
            "metricas": metricas,
            "tipo": tipo,
            "dataset": ruta.parent.name,
        }

    @staticmethod
    def _leer_info(ruta: Path) -> str:
        try:
            return ruta.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""


# --------------------------------------------------------------------------------
# Compuesto
# --------------------------------------------------------------------------------

class RepositorioDeResultados:
    """Vista unificada sobre todas las fuentes de resultados."""

    def __init__(self, fuentes: list[FuenteDeResultados] | None = None) -> None:
        self.fuentes: list[FuenteDeResultados] = fuentes or [
            FuenteWeb(),
            FuenteRepoLegado(),
        ]

    def fuente(self, origen: str) -> FuenteDeResultados:
        for fuente in self.fuentes:
            if fuente.origen == origen:
                return fuente
        raise ResultadoNoEncontrado(f"Origen desconocido: '{origen}'")

    def listar(self, origen: str | None = None) -> list[ResumenExperimento]:
        resumenes: list[ResumenExperimento] = []
        for fuente in self.fuentes:
            if origen and fuente.origen != origen:
                continue
            resumenes.extend(fuente.listar())
        return sorted(resumenes, key=lambda r: r.creado, reverse=True)

    def obtener(self, origen: str, id_experimento: str) -> ResumenExperimento:
        return self.fuente(origen).obtener(id_experimento)

    def comparativa(
        self, origen: str, id_experimento: str, metrica: str
    ) -> dict[str, Any]:
        """Serie de una métrica por optimizador a lo largo de las reglas."""
        resumen = self.obtener(origen, id_experimento)
        series = resumen.metricas.get(metrica, {})
        return {
            "metrica": metrica,
            "reglas": resumen.reglas,
            "series": series,
            "metricas_disponibles": sorted(resumen.metricas),
        }


def _a_float(texto: str) -> float | None:
    try:
        valor = float(texto)
    except (TypeError, ValueError):
        return None
    #JSON no admite NaN ni infinitos.
    return valor if valor == valor and abs(valor) != float("inf") else None


def _valor(bruto: Any) -> Any:
    """Normaliza un valor numérico para CSV y JSON."""
    if bruto is None:
        return ""
    try:
        numero = float(bruto)
    except (TypeError, ValueError):
        return bruto
    if numero != numero or abs(numero) == float("inf"):
        return ""
    return numero


def _json_seguro(objeto: Any) -> Any:
    """Último recurso para tipos que `json` no sabe serializar."""
    return str(objeto)


def _nombre_dataset(exp: DataExperimento) -> str:
    ruta = exp.dataset_path
    if isinstance(ruta, (list, tuple)):
        ruta = ruta[0] if ruta else "dataset"
    return str(ruta).replace("\\", "/").split("/")[-1]

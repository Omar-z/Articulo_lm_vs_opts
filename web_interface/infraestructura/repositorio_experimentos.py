"""Configuraciones de experimento guardadas — patrón Repositorio.

Se almacenan como JSON en `data/configs/<id>.json`, en el mismo formato que come
`rpipeline.py`, de modo que una configuración creada en la web se pueda descargar y
ejecutar con el script sin traducción.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from web_interface.configuracion import DIR_CONFIGS
from web_interface.infraestructura.rutas import ruta_segura, sanear_nombre
from web_interface.presentacion.esquemas import ConfigDTO


class ExperimentoNoEncontrado(KeyError):
    """No existe la configuración pedida."""


class RepositorioDeExperimentos:
    """Alta, consulta, modificación y baja de configuraciones."""

    def __init__(self, directorio: Path | None = None) -> None:
        self.directorio = directorio or DIR_CONFIGS

    def listar(self) -> list[dict[str, Any]]:
        """Fichas ligeras de todas las configuraciones guardadas."""
        fichas: list[dict[str, Any]] = []
        if not self.directorio.exists():
            return fichas
        for archivo in sorted(self.directorio.glob("*.json")):
            try:
                datos = json.loads(archivo.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            experimento = datos.get("experimentacion", {})
            fichas.append(
                {
                    "id": archivo.stem,
                    "nombre": datos.get("_nombre", archivo.stem),
                    "dataset": experimento.get("dataset_path", ""),
                    "tipo": experimento.get("tipo", ""),
                    "reglas": [
                        experimento.get("reglas_inicial"),
                        experimento.get("reglas_total"),
                    ],
                    "optimizadores": [
                        o.get("name", o.get("nombre"))
                        for o in datos.get("optimizadores", [])
                    ],
                    "creado": datos.get("_creado", archivo.stat().st_mtime),
                }
            )
        return sorted(fichas, key=lambda f: f["creado"], reverse=True)

    def obtener(self, id_experimento: str) -> ConfigDTO:
        """Devuelve la configuración ya validada."""
        return ConfigDTO(**self._leer_crudo(id_experimento))

    def obtener_crudo(self, id_experimento: str) -> dict[str, Any]:
        """JSON tal cual, listo para descargar y usar con `rpipeline.py`."""
        crudo = self._leer_crudo(id_experimento)
        return {c: v for c, v in crudo.items() if not c.startswith("_")}

    def guardar(
        self, id_experimento: str, crudo: dict[str, Any], nombre: str | None = None
    ) -> str:
        """Escribe o reemplaza una configuración. Devuelve su identificador."""
        identificador = sanear_nombre(id_experimento, "experimento")
        self.directorio.mkdir(parents=True, exist_ok=True)
        contenido = {
            **crudo,
            "_nombre": nombre or identificador,
            "_creado": time.time(),
        }
        ruta = ruta_segura(self.directorio, f"{identificador}.json")
        ruta.write_text(
            json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return identificador

    def eliminar(self, id_experimento: str) -> None:
        ruta = ruta_segura(self.directorio, f"{sanear_nombre(id_experimento)}.json")
        if not ruta.exists():
            raise ExperimentoNoEncontrado(
                f"No existe el experimento '{id_experimento}'"
            )
        ruta.unlink()

    def _leer_crudo(self, id_experimento: str) -> dict[str, Any]:
        ruta = ruta_segura(self.directorio, f"{sanear_nombre(id_experimento)}.json")
        if not ruta.exists():
            raise ExperimentoNoEncontrado(
                f"No existe el experimento '{id_experimento}'"
            )
        try:
            return json.loads(ruta.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ExperimentoNoEncontrado(
                f"El experimento '{id_experimento}' tiene un JSON inválido: {exc}"
            ) from exc

"""Traducción entre el formulario web y las estructuras del núcleo — patrón Adaptador.

Hay tres representaciones de la misma configuración:

* el **DTO** de pydantic, validado y con tipos concretos, que produce el formulario;
* las **dataclasses** del núcleo (`DataConfig`, `DataExperimento`, `DataOptimizador`),
  sin validación y con rutas relativas al repositorio;
* el **JSON del CLI**, el que come `rpipeline.py`, con la clave `name` en vez de
  `nombre` y sin campos añadidos.

Este ensamblador convierte entre las tres, de modo que una configuración creada en
la web se pueda descargar y ejecutar con el script, y al revés.
"""

from __future__ import annotations

from typing import Any

from web_interface.infraestructura.registros import (
    RegistroDeFuncionesPerdida,
    RegistroDeOptimizadores,
)
from web_interface.infraestructura.rutas import relativizar
from web_interface.nucleo import DataConfig, DataExperimento, DataOptimizador
from web_interface.presentacion.esquemas import (
    ConfigDTO,
    ErrorDeValidacion,
    ExperimentoDTO,
    OptimizadorDTO,
)

#Campos que `DataExperimento` acepta; el DTO tiene alguno más que es solo de la web.
_CAMPOS_EXPERIMENTO = (
    "dataset_path",
    "dataset_header",
    "dataset_sep",
    "dataset_target_col",
    "dataset_map_col",
    "dataset_entradas",
    "dataset_salidas",
    "resultados_path",
    "corridas",
    "epocas",
    "tolerancia",
    "fallos",
    "funcion_perdida",
    "tipo",
    "reglas_inicial",
    "reglas_total",
    "train_size",
    "test_size",
    "val_size",
    "lote_size",
)


class EnsambladorDeConfig:
    """Convierte configuraciones entre sus tres representaciones."""

    def __init__(
        self,
        registro_optimizadores: RegistroDeOptimizadores,
        registro_perdidas: RegistroDeFuncionesPerdida,
    ) -> None:
        self.registro_optimizadores = registro_optimizadores
        self.registro_perdidas = registro_perdidas

    # -- DTO -> núcleo -------------------------------------------------------

    def desde_dto(self, dto: ConfigDTO, semilla_maestra: int | None = None) -> DataConfig:
        experimento = dto.experimentacion
        campos = {
            campo: getattr(experimento, campo)
            for campo in _CAMPOS_EXPERIMENTO
            if hasattr(experimento, campo)
        }
        campos.setdefault("resultados_path", "")
        if campos.get("resultados_path") is None:
            campos["resultados_path"] = ""

        config = DataConfig(
            optimizadores=[
                DataOptimizador(nombre=o.nombre, params=dict(o.params))
                for o in dto.optimizadores
            ],
            experimentos=DataExperimento(**campos),
        )
        #Campo propio de la web: permite reproducir un barrido exacto.
        config.semilla_maestra = semilla_maestra  # type: ignore[attr-defined]
        return config

    # -- núcleo -> DTO -------------------------------------------------------

    def hacia_dto(self, config: DataConfig) -> ConfigDTO:
        exp = config.experimentos
        campos = {
            campo: getattr(exp, campo)
            for campo in _CAMPOS_EXPERIMENTO
            if hasattr(exp, campo)
        }
        return ConfigDTO(
            optimizadores=[
                OptimizadorDTO(nombre=o.nombre, params=dict(o.params))
                for o in config.optimizadores
            ],
            experimentacion=ExperimentoDTO(**campos),
        )

    # -- JSON del CLI --------------------------------------------------------

    def desde_json_cli(self, crudo: dict[str, Any]) -> ConfigDTO:
        """Lee un JSON con el formato de `rpipeline.py` (clave `name`)."""
        return ConfigDTO(
            optimizadores=[
                OptimizadorDTO(nombre=o.get("name", o.get("nombre", "")), params=o.get("params", {}))
                for o in crudo.get("optimizadores", [])
            ],
            experimentacion=ExperimentoDTO(**crudo.get("experimentacion", {})),
        )

    def hacia_json_cli(self, dto: ConfigDTO) -> dict[str, Any]:
        """Produce el JSON exacto que acepta `python rpipeline.py`.

        Las rutas se re-relativizan respecto a la raíz del repositorio, porque el
        script las interpreta desde su directorio de trabajo.
        """
        experimento = dto.experimentacion.model_dump()

        ruta = experimento.get("dataset_path")
        if isinstance(ruta, list):
            experimento["dataset_path"] = [relativizar(r) for r in ruta]
        elif isinstance(ruta, str):
            experimento["dataset_path"] = relativizar(ruta)

        if not experimento.get("resultados_path"):
            nombre = _nombre_corto(experimento["dataset_path"])
            experimento["resultados_path"] = f"resultados/{nombre}/"

        return {
            "optimizadores": [
                {"name": o.nombre, "params": _limpiar_params(o.params)}
                for o in dto.optimizadores
            ],
            "experimentacion": experimento,
        }

    # -- validación ----------------------------------------------------------

    def validar(self, dto: ConfigDTO) -> tuple[list[ErrorDeValidacion], list[str]]:
        """Comprueba lo que pydantic no puede saber: registros y coherencia semántica.

        Se hace antes de encolar porque un nombre de hiperparámetro mal escrito solo
        se manifestaría minutos después, con la CPU ya gastada.
        """
        errores: list[ErrorDeValidacion] = []
        avisos: list[str] = []
        experimento = dto.experimentacion

        if not self.registro_perdidas.existe(experimento.funcion_perdida):
            errores.append(
                ErrorDeValidacion(
                    campo="funcion_perdida",
                    mensaje=(
                        f"'{experimento.funcion_perdida}' no está registrada. "
                        f"Disponibles: {', '.join(self.registro_perdidas.listar())}"
                    ),
                )
            )

        for optimizador in dto.optimizadores:
            campo = f"optimizadores.{optimizador.nombre}"
            if not self.registro_optimizadores.existe(optimizador.nombre):
                errores.append(
                    ErrorDeValidacion(
                        campo=campo,
                        mensaje=f"El optimizador '{optimizador.nombre}' no está registrado",
                    )
                )
                continue

            descriptor = self.registro_optimizadores.obtener(optimizador.nombre)
            if not descriptor.compatible:
                avisos.append(
                    f"{optimizador.nombre}: {descriptor.motivo_incompatible}"
                )

            _, fallos = self.registro_optimizadores.validar_parametros(
                optimizador.nombre, optimizador.params
            )
            errores.extend(
                ErrorDeValidacion(campo=campo, mensaje=mensaje) for mensaje in fallos
            )

        if experimento.tipo == "clasificacion" and experimento.dataset_salidas < 2:
            errores.append(
                ErrorDeValidacion(
                    campo="dataset_salidas",
                    mensaje="Una clasificación necesita al menos dos clases",
                )
            )

        if experimento.lote_size is not None and not experimento.lote_size:
            errores.append(
                ErrorDeValidacion(
                    campo="lote_size", mensaje="El tamaño de lote debe ser mayor que cero"
                )
            )

        #`train_size` está en el esquema del JSON pero el barrido no lo usa.
        proporciones = experimento.proporciones_efectivas()
        if abs(proporciones["entrenamiento"] - experimento.train_size) > 0.01:
            avisos.append(
                "'train_size' es informativo: el reparto real es "
                f"{proporciones['entrenamiento']:.0%} entrenamiento / "
                f"{proporciones['prueba']:.0%} prueba / "
                f"{proporciones['validacion']:.0%} validación, "
                "calculado a partir de 'test_size' y 'val_size' como en el CLI"
            )

        total = experimento.total_entrenamientos() * len(dto.optimizadores)
        if total > 200:
            avisos.append(
                f"Son {total} entrenamientos de hasta {experimento.epocas} épocas; "
                "puede tardar horas"
            )

        return errores, avisos


def _limpiar_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convierte tuplas en listas para que el JSON sea válido."""
    return {
        clave: list(valor) if isinstance(valor, tuple) else valor
        for clave, valor in params.items()
    }


def _nombre_corto(ruta: Any) -> str:
    if isinstance(ruta, list):
        ruta = ruta[0] if ruta else "dataset"
    base = str(ruta).replace("\\", "/").split("/")[-1]
    return base.split(".")[0] or "dataset"

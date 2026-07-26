"""Objetos de transferencia de la API — validación de entrada y salida.

La validación vive aquí para que el dominio reciba datos ya sanos. Los mensajes de
error están en español porque los lee el usuario en el formulario.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OptimizadorDTO(BaseModel):
    """Un optimizador con sus hiperparámetros, tal como lo envía el formulario."""

    nombre: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)

    #Acepta también la forma del JSON del CLI, que usa "name".
    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _admitir_forma_cli(cls, datos: Any) -> Any:
        if isinstance(datos, dict) and "name" in datos and "nombre" not in datos:
            datos = {**datos, "nombre": datos["name"]}
        return datos


class ExperimentoDTO(BaseModel):
    """Configuración completa de un experimento.

    Refleja `DataExperimento` de `funciones_auxiliares.py:257`, añadiendo la
    validación que aquellas dataclasses no tienen.
    """

    #-- dataset
    dataset_path: str | list[str]
    dataset_header: int | dict[str, str] | None = None
    dataset_sep: str | None = None
    dataset_target_col: int | None = None
    dataset_map_col: dict[str, int] | None = None
    dataset_entradas: int = Field(ge=1)
    dataset_salidas: int = Field(ge=1)

    #-- entrenamiento
    corridas: int = Field(default=5, ge=1, le=100)
    epocas: int = Field(default=1000, ge=1, le=1_000_000)
    tolerancia: float = Field(default=1e-12, ge=0)
    fallos: int = Field(default=10, ge=1)
    funcion_perdida: str = "SSE"
    tipo: Literal["regresion", "clasificacion"] = "regresion"
    reglas_inicial: int = Field(default=3, ge=1)
    reglas_total: int = Field(default=5, ge=1)

    #-- particiones
    train_size: float = Field(default=0.6, gt=0, lt=1)
    test_size: float = Field(default=0.2, gt=0, lt=1)
    val_size: float = Field(default=0.2, gt=0, lt=1)
    lote_size: int | None = Field(default=None, ge=1)

    #-- salida
    resultados_path: str | None = None

    @model_validator(mode="after")
    def _coherencia(self) -> "ExperimentoDTO":
        if self.reglas_inicial > self.reglas_total:
            raise ValueError(
                "'reglas_inicial' no puede ser mayor que 'reglas_total'"
            )
        if self.dataset_map_col is not None and self.dataset_target_col is None:
            raise ValueError(
                "'dataset_map_col' requiere que 'dataset_target_col' esté definido"
            )
        if isinstance(self.dataset_path, list) and len(self.dataset_path) != 2:
            raise ValueError(
                "Un dataset dividido debe indicar exactamente dos archivos "
                "(entradas y salidas)"
            )
        return self

    def total_entrenamientos(self) -> int:
        """Cuántas corridas implica esta configuración, sin contar optimizadores."""
        return (self.reglas_total - self.reglas_inicial + 1) * self.corridas

    def proporciones_efectivas(self) -> dict[str, float]:
        """Reparto real de los datos.

        `train_size` no interviene: el CLI aparta `test_size` y subdivide el resto
        con `val_size` (`rpipeline.py:301-302`). Con 0.2/0.2 el reparto real es
        80 % / 16 % / 4 %, no 60/20/20.
        """
        entrenamiento = 1 - self.test_size
        resto = self.test_size
        validacion = resto * self.val_size
        prueba = resto - validacion
        return {
            "entrenamiento": round(entrenamiento, 4),
            "prueba": round(prueba, 4),
            "validacion": round(validacion, 4),
        }


class ConfigDTO(BaseModel):
    """Las dos secciones del JSON de experimento: optimizadores y experimentación."""

    optimizadores: list[OptimizadorDTO] = Field(min_length=1)
    experimentacion: ExperimentoDTO

    @field_validator("optimizadores")
    @classmethod
    def _sin_repetidos(cls, valor: list[OptimizadorDTO]) -> list[OptimizadorDTO]:
        nombres = [o.nombre for o in valor]
        repetidos = {n for n in nombres if nombres.count(n) > 1}
        if repetidos:
            raise ValueError(
                "Hay optimizadores repetidos: " + ", ".join(sorted(repetidos))
            )
        return valor


class SolicitudDeTrabajoDTO(BaseModel):
    """Petición de ejecución de un barrido."""

    tipo: Literal["experimento", "hiperparametros"] = "experimento"
    config: ConfigDTO | None = None
    id_experimento: str | None = None
    dispositivo: str = "cpu"
    minilotes: bool = False
    forzar_float64: bool = False
    semilla_maestra: int | None = None

    #Solo para tipo="hiperparametros": rejilla logarítmica de learning rates.
    valores_hiperparametro: int = Field(default=40, ge=2, le=200)
    lr_max: float = Field(default=0.99, gt=0, lt=1)
    lr_min: float = Field(default=1e-12, gt=0, lt=1)

    @model_validator(mode="after")
    def _origen_de_la_config(self) -> "SolicitudDeTrabajoDTO":
        if self.config is None and self.id_experimento is None:
            raise ValueError(
                "Hay que indicar 'config' o bien 'id_experimento'"
            )
        return self


class ErrorDeValidacion(BaseModel):
    campo: str
    mensaje: str


class RespuestaDeValidacion(BaseModel):
    valido: bool
    errores: list[ErrorDeValidacion] = Field(default_factory=list)
    avisos: list[str] = Field(default_factory=list)
    resumen: dict[str, Any] = Field(default_factory=dict)

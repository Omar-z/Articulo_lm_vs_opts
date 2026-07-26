"""Trabajos encolables — patrón Comando.

La cola necesita objetos autocontenidos que sepan ejecutarse solos. Sin este
patrón, el `GestorDeTrabajos` tendría que conocer la firma de cada tipo de barrido
y ramificar por tipo cada vez que se añade uno nuevo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from web_interface.aplicacion.barridos import (
    BarridoBase,
    BarridoDeExperimento,
    BarridoDeHiperparametros,
)
from web_interface.aplicacion.fachada_anfis import FachadaANFIS
from web_interface.dominio.eventos import SujetoDeProgreso
from web_interface.dominio.politicas_web import PoliticaCancelacion
from web_interface.nucleo import (
    DataConfig,
    PoliticaDeParo,
    PoliticaFallos,
    PoliticaLambdaLM,
    PoliticaNanOrInf,
    PoliticaTolerancia,
)


@dataclass(slots=True)
class ContextoDeEjecucion:
    """Lo que el gestor entrega al comando en el momento de ejecutarlo."""

    id_trabajo: str
    sujeto: SujetoDeProgreso
    cancelacion: PoliticaCancelacion


class Comando(Protocol):
    """Contrato de una unidad de trabajo encolable."""

    id: str
    tipo: str

    def descripcion(self) -> str: ...
    def ejecutar(self, contexto: ContextoDeEjecucion) -> dict[str, Any]: ...


def politicas_del_nucleo(config: DataConfig) -> list[PoliticaDeParo]:
    """Prototipos de las políticas de paro, en el mismo orden que `rpipeline.py:196-206`.

    NaN/Inf va primero para cortar en cuanto la corrida diverge, antes de gastar el
    resto de las épocas. Los parámetros de `PoliticaFallos` se toman del optimizador
    LM si está presente en la configuración, igual que hace el CLI.
    """
    exp = config.experimentos

    lm_params: dict[str, Any] | None = None
    for opt in config.optimizadores:
        if opt.nombre == "LM":
            lm_params = opt.params
            break

    return [
        PoliticaNanOrInf(),
        PoliticaTolerancia(exp.tolerancia),
        PoliticaFallos(
            limite=1e10,
            init=lm_params.get("lambda_init", 0.01) if lm_params else 0.01,
            inc=lm_params.get("lambda_incr", 10) if lm_params else 10,
            dec=lm_params.get("lambda_decr", 0.1) if lm_params else 0.1,
        ),
        PoliticaLambdaLM(),
    ]


@dataclass
class ComandoDeExperimento:
    """Ejecuta un barrido comparativo de optimizadores por número de reglas."""

    id: str
    config: DataConfig
    fachada: FachadaANFIS
    tipo: str = "experimento"
    metadatos: dict[str, Any] = field(default_factory=dict)

    def descripcion(self) -> str:
        exp = self.config.experimentos
        nombre = self.fachada.nombre_dataset(exp)
        optimizadores = ", ".join(o.nombre for o in self.config.optimizadores)
        return (
            f"{nombre} · reglas {exp.reglas_inicial}-{exp.reglas_total} · "
            f"{exp.corridas} corrida(s) · {optimizadores}"
        )

    def crear_barrido(self, contexto: ContextoDeEjecucion) -> BarridoBase:
        return BarridoDeExperimento(
            fachada=self.fachada,
            sujeto=contexto.sujeto,
            cancelacion=contexto.cancelacion,
            prototipos=politicas_del_nucleo(self.config),
            id_trabajo=contexto.id_trabajo,
        )

    def ejecutar(self, contexto: ContextoDeEjecucion) -> dict[str, Any]:
        return self.crear_barrido(contexto).ejecutar(self.config)


@dataclass
class ComandoDeHiperparametros:
    """Busca el mejor learning rate por optimizador y número de reglas."""

    id: str
    config: DataConfig
    fachada: FachadaANFIS
    valores: int = 40
    lr_max: float = 0.99
    lr_min: float = 1e-12
    tipo: str = "hiperparametros"
    metadatos: dict[str, Any] = field(default_factory=dict)

    def descripcion(self) -> str:
        exp = self.config.experimentos
        nombre = self.fachada.nombre_dataset(exp)
        optimizadores = ", ".join(o.nombre for o in self.config.optimizadores)
        return (
            f"Hiperparámetros · {nombre} · reglas {exp.reglas_inicial}-{exp.reglas_total} · "
            f"{self.valores} valores · {optimizadores}"
        )

    def crear_barrido(self, contexto: ContextoDeEjecucion) -> BarridoBase:
        return BarridoDeHiperparametros(
            fachada=self.fachada,
            sujeto=contexto.sujeto,
            cancelacion=contexto.cancelacion,
            prototipos=politicas_del_nucleo(self.config),
            id_trabajo=contexto.id_trabajo,
            valores=self.valores,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
        )

    def ejecutar(self, contexto: ContextoDeEjecucion) -> dict[str, Any]:
        return self.crear_barrido(contexto).ejecutar(self.config)

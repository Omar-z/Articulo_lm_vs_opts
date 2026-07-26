"""Modelos del dominio.

Objetos de datos que viajan entre capas. Ninguno sabe de FastAPI ni del sistema de
archivos: la infraestructura los produce y la presentación los serializa.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch


# --------------------------------------------------------------------------------
# Progreso
# --------------------------------------------------------------------------------

@dataclass(slots=True)
class EventoProgreso:
    """Un hecho ocurrido durante un trabajo, con número de secuencia monotónico.

    La secuencia permite al cliente pedir "todo lo posterior a N" tras una recarga
    de la página o una caída del WebSocket.
    """

    secuencia: int
    id_trabajo: str
    tipo: str  # estado|corrida_inicio|epoca|corrida_fin|regla_fin|bitacora|fin|error
    instante: float
    regla: int | None = None
    optimizador: str | None = None
    corrida: int | None = None
    epoca: int | None = None
    epocas_totales: int | None = None
    loss: float | None = None
    metricas: dict[str, float] | None = None
    fraccion_global: float = 0.0
    datos: dict[str, Any] | None = None

    def a_dict(self) -> dict[str, Any]:
        salida: dict[str, Any] = {
            "secuencia": self.secuencia,
            "id_trabajo": self.id_trabajo,
            "tipo": self.tipo,
            "instante": self.instante,
            "fraccion_global": self.fraccion_global,
        }
        for campo in (
            "regla",
            "optimizador",
            "corrida",
            "epoca",
            "epocas_totales",
            "loss",
            "metricas",
            "datos",
        ):
            valor = getattr(self, campo)
            if valor is not None:
                salida[campo] = valor
        return salida

    def clave_de_curva(self) -> str | None:
        """Identificador de la serie (optimizador, regla, corrida) a la que pertenece."""
        if self.optimizador is None or self.regla is None or self.corrida is None:
            return None
        return f"{self.optimizador}|{self.regla}|{self.corrida}"


# --------------------------------------------------------------------------------
# Optimizadores
# --------------------------------------------------------------------------------

@dataclass(slots=True)
class DescriptorHiperparametro:
    """Un hiperparámetro tal como se le presenta al usuario en el formulario.

    Se deduce por introspección de la firma del constructor del optimizador, así que
    el formulario no necesita mantenerse a mano cuando cambia la versión de PyTorch.
    """

    nombre: str
    tipo: str  # float|int|bool|bool_opcional|tupla|opcion|texto
    por_defecto: Any
    requerido: bool = False
    aridad: int = 1
    opciones: list[str] | None = None
    principal: bool = False

    def a_dict(self) -> dict[str, Any]:
        return {
            "nombre": self.nombre,
            "tipo": self.tipo,
            "por_defecto": self.por_defecto,
            "requerido": self.requerido,
            "aridad": self.aridad,
            "opciones": self.opciones,
            "principal": self.principal,
        }


@dataclass(slots=True)
class DescriptorOptimizador:
    """Un optimizador disponible, con todo lo que la interfaz necesita saber."""

    nombre: str
    clase: type
    origen: str  # torch|propio|plugin
    #True  -> el constructor recibe `params` (los de torch.optim)
    #False -> el constructor recibe el modelo completo (el LM propio)
    recibe_parametros: bool
    hiperparametros: list[DescriptorHiperparametro] = field(default_factory=list)
    doc: str = ""
    modulo: str = ""
    compatible: bool = True
    motivo_incompatible: str | None = None

    def a_dict(self) -> dict[str, Any]:
        return {
            "nombre": self.nombre,
            "origen": self.origen,
            "recibe_parametros": self.recibe_parametros,
            "hiperparametros": [h.a_dict() for h in self.hiperparametros],
            "doc": self.doc,
            "modulo": self.modulo,
            "compatible": self.compatible,
            "motivo_incompatible": self.motivo_incompatible,
        }


# --------------------------------------------------------------------------------
# Datos y corridas
# --------------------------------------------------------------------------------

@dataclass(slots=True)
class ParticionDatos:
    """Los tres subconjuntos ya convertidos a tensores y colocados en su dispositivo."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    semilla: int

    def proporciones(self) -> dict[str, int]:
        return {
            "entrenamiento": int(self.train_x.shape[0]),
            "prueba": int(self.test_x.shape[0]),
            "validacion": int(self.val_x.shape[0]),
        }


@dataclass(slots=True)
class ContextoDeCorrida:
    """Coordenadas de una corrida dentro del barrido, para etiquetar los eventos."""

    regla: int
    optimizador: str
    corrida: int
    epocas: int
    #Etiqueta libre de la variante (p. ej. el learning rate en el barrido de
    #hiperparámetros); vacía en el barrido de experimento.
    variante: str = ""


@dataclass(slots=True)
class ResultadoCorrida:
    """Lo que produce una corrida completa: histórico, métricas finales y motivo de paro."""

    contexto: ContextoDeCorrida
    losses: list[float]
    metricas: dict[str, list[float]]
    evaluacion: dict[str, float]
    motivo_paro: str
    duracion: float
    cancelada: bool = False

    @property
    def epocas_ejecutadas(self) -> int:
        return len(self.losses)

    @property
    def loss_final(self) -> float:
        return self.losses[-1] if self.losses else float("nan")


# --------------------------------------------------------------------------------
# Trabajos
# --------------------------------------------------------------------------------

def nuevo_id() -> str:
    """Identificador corto, ordenable por tiempo y legible en una URL."""
    return f"{int(time.time())}-{uuid.uuid4().hex[:6]}"


@dataclass(slots=True)
class ResumenExperimento:
    """Ficha de un experimento terminado, común a los resultados de la web y a los legados."""

    id: str
    origen: str  # web|repo
    nombre: str
    dataset: str
    tipo: str
    creado: float
    optimizadores: list[str] = field(default_factory=list)
    reglas: list[int] = field(default_factory=list)
    metricas: dict[str, Any] = field(default_factory=dict)
    cancelado: bool = False
    ruta: str = ""

    def a_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "origen": self.origen,
            "nombre": self.nombre,
            "dataset": self.dataset,
            "tipo": self.tipo,
            "creado": self.creado,
            "optimizadores": self.optimizadores,
            "reglas": self.reglas,
            "metricas": self.metricas,
            "cancelado": self.cancelado,
        }

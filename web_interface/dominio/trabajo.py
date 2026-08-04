"""El trabajo: unidad de ejecución de la cola.

Agrupa todo lo que vive mientras dura un barrido — su estado, su bandera de
cancelación, su emisor de eventos, su historial y su bitácora — y delega las
transiciones en el patrón State de `estados.py`.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from web_interface.configuracion import MAX_EVENTOS_HISTORIAL
from web_interface.dominio.estados import (
    EstadoTrabajo,
    Pendiente,
    TransicionInvalida,
)
from web_interface.dominio.eventos import HistorialDeEventos, SujetoDeProgreso


class Trabajo:
    """Un barrido encolado o en ejecución."""

    def __init__(self, id_trabajo: str, tipo: str, descripcion: str) -> None:
        self.id = id_trabajo
        self.tipo = tipo  # experimento|hiperparametros
        self.descripcion = descripcion

        self.estado: EstadoTrabajo = Pendiente()
        self.creado = time.time()
        self.iniciado: float | None = None
        self.terminado: float | None = None

        self.bandera_cancelacion = threading.Event()
        self.sujeto = SujetoDeProgreso(id_trabajo)
        self.historial = HistorialDeEventos(MAX_EVENTOS_HISTORIAL)
        self.sujeto.suscribir(self.historial)

        self.error: str | None = None
        self.resultado: dict[str, Any] | None = None
        self.ruta_resultado: str | None = None
        self.fraccion: float = 0.0
        self.paso_actual: dict[str, Any] = {}
        self.metadatos: dict[str, Any] = {}

    # -- ciclo de vida -------------------------------------------------------

    def transicionar(self, evento: str) -> EstadoTrabajo:
        """Cambia de estado y publica el cambio. Lanza `TransicionInvalida` si no procede."""
        self.estado = self.estado.siguiente(evento)

        if self.estado.nombre == "ejecutando" and self.iniciado is None:
            self.iniciado = time.time()
        elif self.estado.es_terminal():
            self.terminado = time.time()

        self.sujeto.publicar(
            tipo="estado",
            fraccion_global=self.fraccion,
            datos={"estado": self.estado.nombre, "error": self.error},
        )
        return self.estado

    def solicitar_cancelacion(self) -> None:
        """Pide el corte. La política de cancelación lo aplicará en la siguiente época."""
        if not self.estado.puede_cancelar():
            raise TransicionInvalida(
                f"Un trabajo en estado '{self.estado.nombre}' ya no se puede cancelar"
            )
        self.bandera_cancelacion.set()
        self.transicionar("cancelar")

    @property
    def duracion(self) -> float | None:
        if self.iniciado is None:
            return None
        return (self.terminado or time.time()) - self.iniciado

    # -- serialización -------------------------------------------------------

    def a_dict(self, con_paso: bool = True) -> dict[str, Any]:
        datos: dict[str, Any] = {
            "id": self.id,
            "tipo": self.tipo,
            "descripcion": self.descripcion,
            "estado": self.estado.nombre,
            "es_terminal": self.estado.es_terminal(),
            "puede_cancelar": self.estado.puede_cancelar(),
            "puede_eliminar": self.estado.puede_eliminar(),
            "creado": self.creado,
            "iniciado": self.iniciado,
            "terminado": self.terminado,
            "duracion": self.duracion,
            "fraccion": self.fraccion,
            "error": self.error,
            "ruta_resultado": self.ruta_resultado,
            "ultima_secuencia": self.historial.ultima_secuencia,
            "metadatos": self.metadatos,
        }
        if con_paso:
            datos["paso_actual"] = self.paso_actual
        return datos

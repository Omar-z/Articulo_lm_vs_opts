"""Ciclo de vida de un trabajo — patrón State.

Cada endpoint tiene que decidir si su acción es legal: no se puede cancelar un
trabajo terminado, ni eliminar uno que se está ejecutando. Sin este patrón esas
comprobaciones serían cadenas de ``if estado == "..."`` repartidas entre el gestor,
los endpoints y las plantillas.

Transiciones válidas::

    Pendiente ──arrancar──▶ Ejecutando ──terminar──▶ Terminado
        │                       │
        │                       ├──cancelar──▶ Cancelando ──terminar──▶ Cancelado
        │                       └──fallar────▶ Fallido
        └──cancelar──▶ Cancelado
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class TransicionInvalida(Exception):
    """Se pidió una transición que el estado actual no admite."""


@runtime_checkable
class EstadoTrabajo(Protocol):
    """Contrato de un estado del ciclo de vida."""

    nombre: str

    def puede_cancelar(self) -> bool: ...
    def puede_eliminar(self) -> bool: ...
    def es_terminal(self) -> bool: ...
    def siguiente(self, evento: str) -> "EstadoTrabajo": ...


class _EstadoBase:
    """Comportamiento compartido; las subclases declaran su tabla de transiciones."""

    nombre: str = "desconocido"
    _transiciones: dict[str, str] = {}

    def puede_cancelar(self) -> bool:
        return "cancelar" in self._transiciones

    def puede_eliminar(self) -> bool:
        return self.es_terminal()

    def es_terminal(self) -> bool:
        return not self._transiciones

    def siguiente(self, evento: str) -> "EstadoTrabajo":
        destino = self._transiciones.get(evento)
        if destino is None:
            raise TransicionInvalida(
                f"Un trabajo en estado '{self.nombre}' no admite '{evento}'"
            )
        return ESTADOS[destino]()

    def __str__(self) -> str:
        return self.nombre

    def __eq__(self, otro: object) -> bool:
        return isinstance(otro, _EstadoBase) and otro.nombre == self.nombre

    def __hash__(self) -> int:
        return hash(self.nombre)


class Pendiente(_EstadoBase):
    """En la cola, todavía sin arrancar. Cancelarlo lo saca sin ejecutar nada."""

    nombre = "pendiente"
    _transiciones = {"arrancar": "ejecutando", "cancelar": "cancelado", "fallar": "fallido"}

    def puede_eliminar(self) -> bool:
        return True


class Ejecutando(_EstadoBase):
    """El hilo trabajador lo está procesando."""

    nombre = "ejecutando"
    _transiciones = {"cancelar": "cancelando", "terminar": "terminado", "fallar": "fallido"}


class Cancelando(_EstadoBase):
    """Cancelación pedida; se aplicará al terminar la época en curso."""

    nombre = "cancelando"
    _transiciones = {"terminar": "cancelado", "fallar": "fallido"}

    def puede_cancelar(self) -> bool:
        return False


class Terminado(_EstadoBase):
    nombre = "terminado"
    _transiciones = {}


class Cancelado(_EstadoBase):
    """Cortado por el usuario. Los resultados parciales sí se conservan."""

    nombre = "cancelado"
    _transiciones = {}


class Fallido(_EstadoBase):
    nombre = "fallido"
    _transiciones = {}


ESTADOS: dict[str, type[_EstadoBase]] = {
    "pendiente": Pendiente,
    "ejecutando": Ejecutando,
    "cancelando": Cancelando,
    "terminado": Terminado,
    "cancelado": Cancelado,
    "fallido": Fallido,
}


def estado_por_nombre(nombre: str) -> EstadoTrabajo:
    """Reconstruye un estado a partir de su nombre (para deserializar)."""
    clase = ESTADOS.get(nombre)
    if clase is None:
        raise ValueError(f"Estado desconocido: '{nombre}'")
    return clase()

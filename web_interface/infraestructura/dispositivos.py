"""Descubrimiento de los dispositivos de cómputo disponibles.

El modelo ANFIS crea todos sus parámetros en ``torch.float64``
(`V2_Anfis.py:33-35`) y MPS no soporta ese tipo, así que se lista pero marcado
como incompatible: seleccionarlo produciría un `TypeError` al mover el modelo.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from web_interface.configuracion import MOTIVO_MPS_INCOMPATIBLE


@dataclass(slots=True)
class Dispositivo:
    """Un destino de cómputo tal como lo ve la interfaz."""

    id: str
    etiqueta: str
    disponible: bool
    compatible: bool
    motivo: str | None = None

    def a_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "etiqueta": self.etiqueta,
            "disponible": self.disponible,
            "compatible": self.compatible,
            "motivo": self.motivo,
        }


def listar_dispositivos() -> list[Dispositivo]:
    """Devuelve los dispositivos del equipo, con su compatibilidad con el modelo."""
    dispositivos = [
        Dispositivo(id="cpu", etiqueta="CPU", disponible=True, compatible=True)
    ]

    hay_cuda = torch.cuda.is_available()
    dispositivos.append(
        Dispositivo(
            id="cuda",
            etiqueta="CUDA (NVIDIA)",
            disponible=hay_cuda,
            compatible=hay_cuda,
            motivo=None if hay_cuda else "No se detectó ninguna GPU CUDA",
        )
    )

    hay_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    dispositivos.append(
        Dispositivo(
            id="mps",
            etiqueta="MPS (Apple Silicon)",
            disponible=hay_mps,
            compatible=False,
            motivo=(
                MOTIVO_MPS_INCOMPATIBLE
                if hay_mps
                else "No disponible en este equipo"
            ),
        )
    )
    return dispositivos


def dispositivos_utilizables() -> list[str]:
    """Ids de los dispositivos que se pueden seleccionar para entrenar."""
    return [d.id for d in listar_dispositivos() if d.disponible and d.compatible]

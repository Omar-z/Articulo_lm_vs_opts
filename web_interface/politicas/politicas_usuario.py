"""Políticas de paro definidas por el usuario.

Este archivo lo escribe la página `/politicas` de la interfaz web, pero también se
puede editar a mano: al pulsar «Recargar» se vuelve a leer entero.

Cada política es una clase que implementa el mismo contrato que las del núcleo
(`funciones_auxiliares.py:290`):

    class MiPolitica(PoliticaDeParo):
        def __init__(self) -> None:
            self.nombre = "Mi política"

        def apply(self, loss_value: float, optimizador: Any) -> bool:
            return loss_value < 1e-9    # True detiene el entrenamiento

`apply` se llama **una vez por época** desde `train_nfs` y `train_nfs_batch`. La
pérdida llega ya convertida a `float`, aunque el optimizador sea de PyTorch.

Devolver `True` detiene esa corrida; el barrido continúa con la siguiente.
"""

from typing import Any

import numpy as np
import torch  # noqa: F401  (disponible para las políticas que lo necesiten)

from funciones_auxiliares import PoliticaDeParo


class PoliticaPromedioTolerancia(PoliticaDeParo):
    """Ejemplo: detiene cuando el promedio de las últimas épocas baja de la tolerancia.

    A diferencia de `PoliticaTolerancia`, que mira solo la época actual, esta espera
    a que la media de una ventana se estabilice, con lo que un único valor bajo por
    casualidad no corta el entrenamiento.
    """

    def __init__(self, tol: float = 1e-8, ventana: int = 10) -> None:
        self.nombre = "Politica Promedio Tolerancia"
        self.tol = tol
        self.ventana = ventana
        self.historial: list[float] = []

    def apply(self, loss_value: float, optimizador: Any) -> bool:
        self.historial.append(loss_value)
        if len(self.historial) > self.ventana:
            self.historial.pop(0)
        if len(self.historial) < self.ventana:
            return False
        return bool(np.mean(self.historial) < self.tol)




class PoliticaEstancamiento(PoliticaDeParo):
    """Detiene si la pérdida no mejora en N épocas seguidas."""

    def __init__(self, paciencia: int = 25, minimo: float = 1e-6) -> None:
        self.nombre = "Politica Estancamiento"
        self.paciencia = paciencia
        self.minimo = minimo
        self.mejor = float("inf")
        self.sin_mejorar = 0

    def apply(self, loss_value: float, optimizador: Any) -> bool:
        if loss_value < self.mejor - self.minimo:
            self.mejor = loss_value
            self.sin_mejorar = 0
        else:
            self.sin_mejorar += 1
        return self.sin_mejorar >= self.paciencia

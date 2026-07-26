"""Ejemplo de optimizador propio.

Descenso de gradiente con momento y decaimiento exponencial del learning rate.
Sirve para comprobar que el mecanismo de plugins funciona y como plantilla.

Al pulsar «Recargar» en /optimizadores aparece como `SGDConDecaimiento`, con un
formulario de cuatro campos deducido de la firma de `__init__`.
"""

from __future__ import annotations

import torch


class SGDConDecaimiento(torch.optim.Optimizer):
    """SGD con momento y decaimiento exponencial del paso.

    El learning rate efectivo en el paso *t* es ``lr * decaimiento**t``, lo que
    ayuda a que el ajuste fino de los centros y sigmas de las funciones de
    pertenencia no oscile al final del entrenamiento.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        decaimiento: float = 0.999,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"El learning rate debe ser positivo, no {lr}")
        if not 0.0 < decaimiento <= 1.0:
            raise ValueError(f"El decaimiento debe estar en (0, 1], no {decaimiento}")

        super().__init__(
            params,
            {
                "lr": lr,
                "momentum": momentum,
                "decaimiento": decaimiento,
                "weight_decay": weight_decay,
            },
        )

    @torch.no_grad()
    def step(self, closure=None):
        perdida = None
        if closure is not None:
            with torch.enable_grad():
                perdida = closure()

        for grupo in self.param_groups:
            for parametro in grupo["params"]:
                if parametro.grad is None:
                    continue

                gradiente = parametro.grad
                if grupo["weight_decay"]:
                    gradiente = gradiente.add(parametro, alpha=grupo["weight_decay"])

                estado = self.state[parametro]
                if not estado:
                    estado["paso"] = 0
                    #Mismo dtype que el parámetro: el modelo ANFIS es float64.
                    estado["velocidad"] = torch.zeros_like(parametro)

                estado["paso"] += 1
                velocidad = estado["velocidad"]
                velocidad.mul_(grupo["momentum"]).add_(gradiente)

                paso_efectivo = grupo["lr"] * grupo["decaimiento"] ** estado["paso"]
                parametro.add_(velocidad, alpha=-paso_efectivo)

        return perdida

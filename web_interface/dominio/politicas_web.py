"""Políticas propias de la interfaz — patrón Estrategia.

**Este es el pivote de toda la arquitectura.** `train_nfs` y `train_nfs_batch` no
exponen ningún callback, pero construyen un `CompositorDePoliticas` con la lista
`early_stop` y lo consultan en cada época (`V2_Anfis.py:464` y `:554`):

    if(politicas_de_paro.apply(loss,optimizer)):

Inyectando en esa lista estrategias que cumplan el mismo `Protocol` que las del
núcleo se obtiene progreso época a época y cancelación cooperativa **sin modificar
una sola línea de los scripts**.

Orden obligatorio de la lista::

    [observador, cancelacion, *deepcopy(politicas_del_nucleo)]

El observador va primero porque siempre devuelve ``False`` y así ve todas las
épocas: `CompositorDePoliticas` corta en cuanto una política devuelve ``True``. La
cancelación va segunda para que se le atribuya el motivo del paro. Las del núcleo
quedan al final y siguen recibiendo el flujo íntegro, que es lo que necesita
`PoliticaFallos` para su contador.

Heredan de `PoliticaDeParo` igual que las del núcleo, para reutilizar su `__str__`:
`CompositorDePoliticas` imprime la política que disparó el paro.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import torch

from web_interface.dominio.eventos import SujetoDeProgreso
from web_interface.dominio.modelos import ContextoDeCorrida
from web_interface.nucleo import PoliticaDeParo


def _a_float(valor: Any) -> float:
    """Normaliza la pérdida que entrega el bucle de entrenamiento.

    Los optimizadores de PyTorch pasan un `torch.Tensor` con grafo asociado y el LM
    propio pasa un `float`. Se hace `detach()` porque convertir directamente un
    tensor con `requires_grad=True` emite un `UserWarning` en cada época.
    """
    if isinstance(valor, torch.Tensor):
        return float(valor.detach())
    return float(valor)


class PoliticaObservador(PoliticaDeParo):
    """Publica el progreso de cada época. Nunca detiene el entrenamiento.

    Registra internamente **todas** las pérdidas, pero solo publica una de cada
    `cada_n` para no inundar el WebSocket: un barrido completo puede rozar los
    240 000 eventos de época. La última siempre se publica.
    """

    def __init__(
        self,
        sujeto: SujetoDeProgreso,
        contexto: ContextoDeCorrida,
        cada_n: int = 1,
        fraccion: Callable[[int], float] | None = None,
    ) -> None:
        self.nombre = "Observador de progreso"
        self.sujeto = sujeto
        self.contexto = contexto
        self.cada_n = max(1, cada_n)
        self.fraccion = fraccion
        self.epoca = 0
        self.losses: list[float] = []

    def apply(self, loss_value: Any, optimizador: Any) -> bool:
        self.epoca += 1
        valor = _a_float(loss_value)
        self.losses.append(valor)

        es_ultima = self.epoca >= self.contexto.epocas
        if self.epoca % self.cada_n == 0 or self.epoca == 1 or es_ultima:
            self.sujeto.publicar(
                tipo="epoca",
                regla=self.contexto.regla,
                optimizador=self.contexto.optimizador,
                corrida=self.contexto.corrida,
                epoca=self.epoca,
                epocas_totales=self.contexto.epocas,
                loss=valor,
                fraccion_global=self.fraccion(self.epoca) if self.fraccion else 0.0,
            )
        return False


class PoliticaNormalizada(PoliticaDeParo):
    """Adaptador de tipo entre el bucle de entrenamiento y las políticas del núcleo.

    Las políticas de `funciones_auxiliares` se escribieron asumiendo que la pérdida
    llega como `float`, que es lo que entrega el optimizador LM. Pero con los
    optimizadores de PyTorch `train_nfs` pasa el `torch.Tensor` con grafo asociado
    (`V2_Anfis.py:440` y `:464`). Esa diferencia ya provocó un fallo real en
    `PoliticaNanOrInf`::

        np.isnan(tensor) -> Tensor.__array__ -> .numpy()
        RuntimeError: Can't call numpy() on Tensor that requires grad

    Envolviendo cada política del núcleo en este adaptador se convierte la pérdida a
    `float` una sola vez antes de delegar, de modo que la web funciona con
    cualquier política que se añada al núcleo, la escriba quien la escriba. De paso
    evita que `PoliticaFallos` conserve tensores en `best_loss` de una época a otra.

    Se delegan `nombre` y `__str__` para que el mensaje que imprime
    `CompositorDePoliticas` al detener siga identificando a la política real.
    """

    def __init__(self, politica: Any) -> None:
        self.politica = politica

    @property
    def nombre(self) -> str:
        return getattr(self.politica, "nombre", type(self.politica).__name__)

    def apply(self, loss_value: Any, optimizador: Any) -> bool:
        return bool(self.politica.apply(_a_float(loss_value), optimizador))

    def __str__(self) -> str:
        return str(self.politica)


class PoliticaCancelacion(PoliticaDeParo):
    """Corta el entrenamiento cuando el usuario pulsa Cancelar.

    La bandera se comparte **por referencia** entre el hilo del servidor y el del
    entrenamiento; por eso esta política nunca se copia con `deepcopy` (duplicar el
    `Event` la dejaría sorda).
    """

    def __init__(self, bandera: threading.Event) -> None:
        self.nombre = "Cancelación solicitada por el usuario"
        self.bandera = bandera

    def apply(self, loss_value: Any, optimizador: Any) -> bool:
        return self.bandera.is_set()


class PoliticaLimiteDeTiempo(PoliticaDeParo):
    """Corta una corrida que excede su presupuesto de tiempo.

    Evita que una combinación mal condicionada (p. ej. un learning rate absurdo con
    muchas reglas) bloquee la cola indefinidamente.
    """

    def __init__(self, segundos: float) -> None:
        self.nombre = f"Límite de tiempo ({segundos:.0f} s)"
        self.segundos = segundos
        self.inicio = time.monotonic()

    def reiniciar(self) -> None:
        self.inicio = time.monotonic()

    def apply(self, loss_value: Any, optimizador: Any) -> bool:
        return (time.monotonic() - self.inicio) > self.segundos

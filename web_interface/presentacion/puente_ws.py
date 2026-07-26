"""Puente entre el hilo trabajador y el event loop de asyncio.

Los eventos nacen en el hilo de entrenamiento y hay que entregarlos a los
WebSockets, que viven en el event loop. Se usa ``loop.call_soon_threadsafe`` y no
``asyncio.run_coroutine_threadsafe`` porque este último crea una corrutina y un
`Future` por evento, y el volumen es alto: un barrido completo de 8 reglas × 6
optimizadores × 5 corridas × 1000 épocas roza los 240 000 eventos.

Cada suscriptor tiene su propia cola acotada con descarte del más antiguo, de modo
que un navegador lento nunca frene el entrenamiento.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from web_interface.configuracion import MAX_COLA_SUSCRIPTOR
from web_interface.dominio.eventos import EventoProgreso


class PuenteDeEventos:
    """Reparte los eventos de progreso a los WebSockets suscritos."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._suscriptores: dict[str, set[asyncio.Queue]] = {}

    def enlazar(self, loop: asyncio.AbstractEventLoop) -> None:
        """Asocia el puente al event loop. Se llama al arrancar la aplicación."""
        self._loop = loop

    def desenlazar(self) -> None:
        self._loop = None
        self._suscriptores.clear()

    # -- lado del hilo trabajador --------------------------------------------

    def notificar(self, evento: EventoProgreso) -> None:
        """Punto de entrada desde el hilo de entrenamiento. No bloquea."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._repartir, evento)
        except RuntimeError:
            #El loop se cerró entre la comprobación y la llamada: se descarta.
            pass

    # -- lado del event loop -------------------------------------------------

    def _repartir(self, evento: EventoProgreso) -> None:
        for cola in list(self._suscriptores.get(evento.id_trabajo, ())):
            try:
                cola.put_nowait(evento)
            except asyncio.QueueFull:
                #Cliente lento: se tira el evento más viejo y entra el nuevo.
                try:
                    cola.get_nowait()
                    cola.put_nowait(evento)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    @asynccontextmanager
    async def suscripcion(self, id_trabajo: str) -> AsyncIterator[asyncio.Queue]:
        """Alta y baja garantizadas de un suscriptor."""
        cola: asyncio.Queue = asyncio.Queue(maxsize=MAX_COLA_SUSCRIPTOR)
        self._suscriptores.setdefault(id_trabajo, set()).add(cola)
        try:
            yield cola
        finally:
            suscriptores = self._suscriptores.get(id_trabajo)
            if suscriptores is not None:
                suscriptores.discard(cola)
                if not suscriptores:
                    del self._suscriptores[id_trabajo]

    def cuantos_suscriptores(self, id_trabajo: str) -> int:
        return len(self._suscriptores.get(id_trabajo, ()))

"""WebSocket de progreso.

Al conectar se envía un **snapshot** con el estado completo del trabajo y las
curvas ya submuestreadas; a partir de ahí solo llegan deltas. Así una recarga de la
página (o una reconexión tras perder la red) reconstruye la vista sin depender de
haber estado escuchando desde el principio.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from web_interface.aplicacion.gestor_trabajos import TrabajoNoEncontrado
from web_interface.configuracion import MAX_PUNTOS_CURVA

router = APIRouter()

#Códigos de cierre del protocolo WebSocket
_CIERRE_NORMAL = 1000
_CIERRE_NO_ENCONTRADO = 4004


@router.websocket("/ws/trabajos/{id_trabajo}")
async def progreso_de_trabajo(websocket: WebSocket, id_trabajo: str) -> None:
    estado = websocket.app.state
    gestor = estado.gestor
    puente = estado.puente

    await websocket.accept()

    try:
        trabajo = gestor.obtener(id_trabajo)
    except TrabajoNoEncontrado:
        await websocket.send_json(
            {"tipo": "error", "datos": {"mensaje": f"No existe el trabajo '{id_trabajo}'"}}
        )
        await websocket.close(code=_CIERRE_NO_ENCONTRADO)
        return

    async with puente.suscripcion(id_trabajo) as cola:
        #El snapshot se compone DESPUÉS de suscribirse: si llegan eventos entre
        #medias, el cliente los descartará por número de secuencia.
        await websocket.send_json(_snapshot(trabajo, gestor))

        recepcion = asyncio.create_task(_escuchar(websocket, trabajo, gestor))
        try:
            while True:
                if recepcion.done():
                    break
                try:
                    evento = await asyncio.wait_for(cola.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    #Latido: mantiene viva la conexión a través de proxies.
                    await websocket.send_json(
                        {"tipo": "latido", "estado": trabajo.estado.nombre}
                    )
                    continue

                await websocket.send_json(evento.a_dict())

                if evento.tipo in ("fin", "estado") and trabajo.estado.es_terminal():
                    await websocket.send_json(_snapshot(trabajo, gestor, final=True))
                    break
        except WebSocketDisconnect:
            pass
        finally:
            recepcion.cancel()

    try:
        await websocket.close(code=_CIERRE_NORMAL)
    except RuntimeError:
        pass


async def _escuchar(websocket: WebSocket, trabajo: Any, gestor: Any) -> None:
    """Atiende las peticiones del cliente (repetición de eventos, ping)."""
    while True:
        mensaje = await websocket.receive_json()
        accion = mensaje.get("accion")

        if accion == "replay":
            desde = int(mensaje.get("desde", 0))
            for evento in trabajo.historial.desde(desde):
                await websocket.send_json(evento.a_dict())
        elif accion == "ping":
            await websocket.send_json({"tipo": "pong"})
        elif accion == "snapshot":
            await websocket.send_json(_snapshot(trabajo, gestor))


def _snapshot(trabajo: Any, gestor: Any, final: bool = False) -> dict[str, Any]:
    """Estado completo del trabajo, suficiente para pintar el monitor desde cero."""
    return {
        "tipo": "snapshot",
        "final": final,
        "trabajo": trabajo.a_dict(),
        "secuencia": trabajo.historial.ultima_secuencia,
        "curvas": trabajo.historial.curvas(MAX_PUNTOS_CURVA),
        "corridas_terminadas": trabajo.historial.corridas_terminadas(),
        "bitacora": gestor.bitacora(trabajo.id, 200),
    }


@router.websocket("/ws/cola")
async def progreso_de_cola(websocket: WebSocket) -> None:
    """Vista de la cola: se refresca cada vez que cambia algún estado."""
    estado = websocket.app.state
    gestor = estado.gestor

    await websocket.accept()
    try:
        anterior: list[dict[str, Any]] = []
        while True:
            actual = [t.a_dict(con_paso=False) for t in gestor.listar()]
            if actual != anterior:
                await websocket.send_json({"tipo": "cola", "trabajos": actual})
                anterior = actual
            await asyncio.sleep(1.0)
    except (WebSocketDisconnect, RuntimeError):
        pass

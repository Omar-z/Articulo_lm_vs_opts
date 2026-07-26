"""API de trabajos: encolar, consultar, cancelar y recuperar progreso."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from web_interface.aplicacion.comandos import (
    ComandoDeExperimento,
    ComandoDeHiperparametros,
)
from web_interface.aplicacion.gestor_trabajos import TrabajoNoEncontrado
from web_interface.configuracion import MAX_PUNTOS_CURVA
from web_interface.dominio.estados import TransicionInvalida
from web_interface.dominio.modelos import nuevo_id
from web_interface.presentacion.esquemas import (
    ConfigDTO,
    RespuestaDeValidacion,
    SolicitudDeTrabajoDTO,
)

router = APIRouter(prefix="/api/trabajos", tags=["trabajos"])


def _servicios(request: Request) -> Any:
    return request.app.state


def _resolver_config(estado: Any, solicitud: SolicitudDeTrabajoDTO) -> ConfigDTO:
    """Obtiene la configuración de la petición o del repositorio de experimentos."""
    if solicitud.config is not None:
        return solicitud.config

    repositorio = getattr(estado, "repositorio_experimentos", None)
    if repositorio is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Todavía no hay experimentos guardados; envía 'config' directamente",
        )
    try:
        return repositorio.obtener(solicitud.id_experimento)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No existe el experimento '{solicitud.id_experimento}'",
        ) from exc


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def crear_trabajo(
    solicitud: SolicitudDeTrabajoDTO, request: Request
) -> dict[str, Any]:
    """Encola un barrido. Responde de inmediato, sin esperar al entrenamiento."""
    estado = _servicios(request)
    config_dto = _resolver_config(estado, solicitud)

    errores, avisos = estado.ensamblador.validar(config_dto)
    if errores:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "mensaje": "La configuración tiene errores",
                "errores": [e.model_dump() for e in errores],
            },
        )

    dispositivo = _validar_dispositivo(estado, solicitud.dispositivo)
    config = estado.ensamblador.desde_dto(config_dto, solicitud.semilla_maestra)

    fachada = estado.crear_fachada(
        dispositivo=dispositivo,
        minilotes=solicitud.minilotes,
        forzar_float64=solicitud.forzar_float64,
    )
    if solicitud.tipo == "hiperparametros":
        comando: Any = ComandoDeHiperparametros(
            id=nuevo_id(),
            config=config,
            fachada=fachada,
            valores=solicitud.valores_hiperparametro,
            lr_max=solicitud.lr_max,
            lr_min=solicitud.lr_min,
        )
    else:
        comando = ComandoDeExperimento(id=nuevo_id(), config=config, fachada=fachada)

    trabajo = estado.gestor.encolar(
        comando,
        metadatos={
            "dispositivo": dispositivo,
            "minilotes": solicitud.minilotes,
            "config": config_dto.model_dump(),
            "avisos": avisos,
        },
    )

    return {
        "id_trabajo": trabajo.id,
        "estado": trabajo.estado.nombre,
        "descripcion": trabajo.descripcion,
        "posicion_en_cola": estado.gestor.posicion_en_cola(trabajo.id),
        "avisos": avisos,
    }


def _validar_dispositivo(estado: Any, pedido: str) -> str:
    from web_interface.infraestructura.dispositivos import listar_dispositivos

    for dispositivo in listar_dispositivos():
        if dispositivo.id == pedido:
            if not dispositivo.disponible:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"El dispositivo '{pedido}' no está disponible en este equipo",
                )
            if not dispositivo.compatible:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"No se puede entrenar en '{pedido}': {dispositivo.motivo}",
                )
            return pedido
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Dispositivo desconocido: '{pedido}'",
    )


@router.post("/validar", response_model=RespuestaDeValidacion)
async def validar_config(config: ConfigDTO, request: Request) -> RespuestaDeValidacion:
    """Comprueba una configuración sin ejecutarla."""
    estado = _servicios(request)
    errores, avisos = estado.ensamblador.validar(config)
    experimento = config.experimentacion
    return RespuestaDeValidacion(
        valido=not errores,
        errores=errores,
        avisos=avisos,
        resumen={
            "total_entrenamientos": experimento.total_entrenamientos()
            * len(config.optimizadores),
            "proporciones_efectivas": experimento.proporciones_efectivas(),
            "epocas_maximas": experimento.epocas,
        },
    )


@router.get("")
async def listar_trabajos(request: Request, estado_filtro: str | None = None) -> list[dict[str, Any]]:
    gestor = _servicios(request).gestor
    return [t.a_dict() for t in gestor.listar(estado_filtro)]


@router.get("/{id_trabajo}")
async def obtener_trabajo(id_trabajo: str, request: Request) -> dict[str, Any]:
    return _buscar(request, id_trabajo).a_dict()


@router.post("/{id_trabajo}/cancelar")
async def cancelar_trabajo(id_trabajo: str, request: Request) -> dict[str, Any]:
    """Pide el corte. Se aplica al terminar la época en curso."""
    gestor = _servicios(request).gestor
    try:
        return gestor.cancelar(id_trabajo).a_dict()
    except TrabajoNoEncontrado as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TransicionInvalida as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/{id_trabajo}", status_code=status.HTTP_204_NO_CONTENT)
async def eliminar_trabajo(id_trabajo: str, request: Request) -> None:
    gestor = _servicios(request).gestor
    try:
        gestor.eliminar(id_trabajo)
    except TrabajoNoEncontrado as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{id_trabajo}/eventos")
async def eventos_del_trabajo(
    id_trabajo: str, request: Request, desde: int = 0, limite: int = 2000
) -> dict[str, Any]:
    """Eventos posteriores a `desde`. Es el plan B si el WebSocket no está disponible."""
    trabajo = _buscar(request, id_trabajo)
    eventos = trabajo.historial.desde(desde, limite)
    return {
        "eventos": [e.a_dict() for e in eventos],
        "ultima_secuencia": trabajo.historial.ultima_secuencia,
        "estado": trabajo.estado.nombre,
    }


@router.get("/{id_trabajo}/bitacora")
async def bitacora_del_trabajo(
    id_trabajo: str, request: Request, ultimas: int = 200
) -> dict[str, Any]:
    """Salida por consola del núcleo, ya sin códigos ANSI."""
    gestor = _servicios(request).gestor
    _buscar(request, id_trabajo)
    return {"lineas": gestor.bitacora(id_trabajo, ultimas)}


@router.get("/{id_trabajo}/curva")
async def curva_del_trabajo(
    id_trabajo: str,
    request: Request,
    optimizador: str,
    regla: int,
    corrida: int = 0,
    puntos: int = MAX_PUNTOS_CURVA,
) -> dict[str, Any]:
    """Curva de pérdida de una combinación concreta, submuestreada."""
    trabajo = _buscar(request, id_trabajo)
    clave = f"{optimizador}|{regla}|{corrida}"
    return {"clave": clave, **trabajo.historial.curva(clave, puntos)}


def _buscar(request: Request, id_trabajo: str):
    try:
        return _servicios(request).gestor.obtener(id_trabajo)
    except TrabajoNoEncontrado as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

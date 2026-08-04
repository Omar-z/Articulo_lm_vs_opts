"""API de políticas de paro: catálogo, plantilla, validación, prueba y guardado."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from web_interface.infraestructura.repositorio_politicas import (
    EscrituraNoPermitida,
    PoliticaRechazada,
    nombre_a_clase,
    plantilla_de_politica,
)

router = APIRouter(prefix="/api/politicas", tags=["politicas"])


class PoliticaDTO(BaseModel):
    """Lo que envía el editor: el nombre legible y el código de la clase."""

    nombre: str = Field(min_length=1, max_length=120)
    codigo: str = Field(min_length=1, max_length=200_000)
    #Si no se indica, se deduce del nombre.
    nombre_clase: str | None = None


def _servicio(request: Request):
    return request.app.state.servicio_politicas


def _clase_de(dto: PoliticaDTO) -> str:
    if dto.nombre_clase:
        return dto.nombre_clase
    try:
        return nombre_a_clase(dto.nombre)
    except PoliticaRechazada as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("")
async def listar(request: Request) -> dict[str, Any]:
    """Catálogo completo: las cuatro del núcleo y las del usuario."""
    servicio = _servicio(request)
    return {
        "politicas": [p.a_dict() for p in servicio.listar()],
        "error_de_carga": servicio.error_de_carga,
        "archivo": str(servicio.repositorio.archivo),
        "escritura_permitida": servicio.repositorio.permitir_escritura,
    }


@router.get("/plantilla")
async def plantilla(nombre: str = "") -> dict[str, str]:
    """Esqueleto de clase que el editor muestra por defecto."""
    try:
        clase = nombre_a_clase(nombre) if nombre.strip() else "MiPolitica"
    except PoliticaRechazada as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"nombre_clase": clase, "codigo": plantilla_de_politica(nombre)}


@router.get("/archivo", response_class=PlainTextResponse)
async def archivo_completo(request: Request) -> str:
    """Contenido íntegro de `politicas_usuario.py`, por si se quiere revisar."""
    return _servicio(request).repositorio.codigo_completo()


@router.post("/recargar")
async def recargar(request: Request) -> dict[str, Any]:
    """Vuelve a leer el archivo, por si se editó a mano."""
    return _servicio(request).recargar()


@router.post("/validar")
async def validar(dto: PoliticaDTO, request: Request) -> dict[str, Any]:
    """Comprueba sintaxis y contrato, y ejecuta la política con pérdidas de prueba.

    No escribe nada: sirve para que el usuario itere en el editor sin arriesgarse a
    dejar el archivo en un estado que impida cargar el resto de políticas.
    """
    servicio = _servicio(request)
    clase = _clase_de(dto)

    problemas = servicio.validar(clase, dto.codigo)
    if problemas:
        return {"valido": False, "nombre_clase": clase, "errores": problemas}

    try:
        prueba = servicio.probar(clase, dto.codigo)
    except PoliticaRechazada as exc:
        return {"valido": False, "nombre_clase": clase, "errores": [str(exc)]}

    avisos: list[str] = []
    if prueba["nunca_detiene"]:
        avisos.append(
            "Con las pérdidas de prueba no se detiene nunca: el entrenamiento "
            "llegará siempre al máximo de épocas"
        )
    if prueba["detiene_siempre"]:
        avisos.append(
            "Se detiene ya en la primera época de prueba: revisa la condición o "
            "las corridas terminarán de inmediato"
        )

    return {
        "valido": True,
        "nombre_clase": clase,
        "errores": [],
        "avisos": avisos,
        "prueba": prueba,
    }


@router.post("", status_code=status.HTTP_201_CREATED)
async def guardar(dto: PoliticaDTO, request: Request) -> dict[str, Any]:
    """Añade o reemplaza la política en el archivo, tras validarla y probarla."""
    servicio = _servicio(request)
    clase = _clase_de(dto)
    try:
        politica = servicio.guardar(clase, dto.codigo)
    except EscrituraNoPermitida as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except PoliticaRechazada as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return politica.a_dict()


@router.delete("/{nombre_clase}", status_code=status.HTTP_204_NO_CONTENT)
async def eliminar(nombre_clase: str, request: Request) -> None:
    servicio = _servicio(request)
    try:
        servicio.eliminar(nombre_clase)
    except EscrituraNoPermitida as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

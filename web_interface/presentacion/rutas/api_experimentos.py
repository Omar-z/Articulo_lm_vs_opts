"""API de configuraciones de experimento: guardar, recuperar y exportar al CLI."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from web_interface.dominio.modelos import nuevo_id
from web_interface.infraestructura.repositorio_experimentos import (
    ExperimentoNoEncontrado,
)
from web_interface.infraestructura.rutas import RutaNoPermitida
from web_interface.presentacion.esquemas import ConfigDTO

router = APIRouter(prefix="/api/experimentos", tags=["experimentos"])


def _repo(request: Request):
    return request.app.state.repositorio_experimentos


def _ensamblador(request: Request):
    return request.app.state.ensamblador


@router.get("")
async def listar(request: Request) -> list[dict[str, Any]]:
    return _repo(request).listar()


@router.post("", status_code=status.HTTP_201_CREATED)
async def crear(
    config: ConfigDTO, request: Request, nombre: str | None = None
) -> dict[str, Any]:
    """Guarda una configuración en el formato que acepta `rpipeline.py`."""
    crudo = _ensamblador(request).hacia_json_cli(config)
    identificador = _repo(request).guardar(nuevo_id(), crudo, nombre)
    return {"id": identificador, "config": crudo}


@router.get("/{id_experimento}")
async def obtener(id_experimento: str, request: Request) -> dict[str, Any]:
    try:
        return _repo(request).obtener(id_experimento).model_dump()
    except (ExperimentoNoEncontrado, RutaNoPermitida) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{id_experimento}")
async def actualizar(
    id_experimento: str, config: ConfigDTO, request: Request, nombre: str | None = None
) -> dict[str, Any]:
    crudo = _ensamblador(request).hacia_json_cli(config)
    _repo(request).guardar(id_experimento, crudo, nombre)
    return {"id": id_experimento, "config": crudo}


@router.delete("/{id_experimento}", status_code=status.HTTP_204_NO_CONTENT)
async def eliminar(id_experimento: str, request: Request) -> None:
    try:
        _repo(request).eliminar(id_experimento)
    except (ExperimentoNoEncontrado, RutaNoPermitida) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{id_experimento}/json-cli")
async def exportar_para_cli(id_experimento: str, request: Request) -> JSONResponse:
    """Descarga el JSON listo para `python rpipeline.py <archivo>`."""
    try:
        crudo = _repo(request).obtener_crudo(id_experimento)
    except (ExperimentoNoEncontrado, RutaNoPermitida) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return JSONResponse(
        content=crudo,
        headers={
            "Content-Disposition": f'attachment; filename="{id_experimento}_tests.json"'
        },
    )


@router.post("/exportar")
async def exportar_config(config: ConfigDTO, request: Request) -> JSONResponse:
    """Convierte una configuración del formulario al JSON del CLI, sin guardarla."""
    crudo = _ensamblador(request).hacia_json_cli(config)
    return JSONResponse(
        content=crudo,
        headers={"Content-Disposition": 'attachment; filename="experimento_tests.json"'},
    )


@router.post("/importar")
async def importar_config(request: Request) -> dict[str, Any]:
    """Lee un JSON con el formato del CLI y lo devuelve como configuración del formulario."""
    try:
        crudo = json.loads((await request.body()).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"JSON inválido: {exc}") from exc

    try:
        dto = _ensamblador(request).desde_json_cli(crudo)
    except Exception as exc:  # noqa: BLE001 - el error de pydantic ya es descriptivo
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return dto.model_dump()

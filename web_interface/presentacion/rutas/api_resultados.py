"""API de resultados: historial, comparativas y descargas."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from web_interface.infraestructura.repositorio_resultados import (
    FuenteRepoLegado,
    ResultadoNoEncontrado,
)
from web_interface.infraestructura.rutas import RutaNoPermitida

router = APIRouter(prefix="/api/resultados", tags=["resultados"])


def _repo(request: Request):
    return request.app.state.repositorio_resultados


@router.get("")
async def listar(request: Request, origen: str | None = None) -> list[dict[str, Any]]:
    """Todos los experimentos: los de la web y los que dejó el script."""
    return [r.a_dict() for r in _repo(request).listar(origen)]


@router.get("/{origen}/{id_experimento}")
async def detalle(origen: str, id_experimento: str, request: Request) -> dict[str, Any]:
    """Ficha del experimento. Nunca abre el `resultados.json` grande."""
    try:
        return _repo(request).obtener(origen, id_experimento).a_dict()
    except (ResultadoNoEncontrado, RutaNoPermitida) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{origen}/{id_experimento}/comparativa")
async def comparativa(
    origen: str, id_experimento: str, request: Request, metrica: str = "loss"
) -> dict[str, Any]:
    """Valor de una métrica por optimizador a lo largo del rango de reglas."""
    try:
        return _repo(request).comparativa(origen, id_experimento, metrica)
    except (ResultadoNoEncontrado, RutaNoPermitida) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{origen}/{id_experimento}/csv")
async def descargar_csv(
    origen: str, id_experimento: str, request: Request
) -> FileResponse:
    try:
        ruta = _repo(request).fuente(origen).ruta_csv(id_experimento)
    except (ResultadoNoEncontrado, RutaNoPermitida) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not ruta.exists():
        raise HTTPException(status_code=404, detail="No hay CSV para este experimento")
    return FileResponse(
        ruta, media_type="text/csv", filename=f"{id_experimento}_resultados.csv"
    )


@router.get("/repo/{id_experimento}/graficas")
async def graficas_legadas(id_experimento: str, request: Request) -> list[dict[str, str]]:
    """PNG que generó `rpipeline.py` con la opción `graficar`."""
    fuente = _repo(request).fuente("repo")
    if not isinstance(fuente, FuenteRepoLegado):
        return []
    try:
        return fuente.graficas(id_experimento)
    except RutaNoPermitida as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/repo/{id_experimento}/grafica/{grupo}/{nombre}")
async def grafica_legada(
    id_experimento: str, grupo: str, nombre: str, request: Request
) -> FileResponse:
    """Sirve un PNG del directorio `resultados/`, en solo lectura."""
    fuente = _repo(request).fuente("repo")
    if not isinstance(fuente, FuenteRepoLegado):
        raise HTTPException(status_code=404, detail="Origen no disponible")
    try:
        ruta = fuente.ruta_grafica(id_experimento, grupo, nombre)
    except RutaNoPermitida as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not ruta.exists() or ruta.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="No existe esa gráfica")
    return FileResponse(ruta, media_type="image/png")

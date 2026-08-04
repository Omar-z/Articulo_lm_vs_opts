"""API del catálogo de optimizadores y funciones de pérdida."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from web_interface.configuracion import PERMITIR_PLUGINS
from web_interface.infraestructura.registros import OptimizadorDesconocido

router = APIRouter(prefix="/api", tags=["optimizadores"])


class ValidacionOptimizadorDTO(BaseModel):
    nombre: str
    params: dict[str, Any] = Field(default_factory=dict)


def _servicio(request: Request):
    return request.app.state.servicio_optimizadores


@router.get("/optimizadores")
async def listar(request: Request, origen: str | None = None) -> dict[str, Any]:
    """Catálogo completo: los de `torch.optim`, el LM propio y los plugins."""
    servicio = _servicio(request)
    return {
        "optimizadores": [d.a_dict() for d in servicio.listar(origen)],
        "errores_plugins": [e.a_dict() for e in servicio.ultimos_errores],
        "plugins_permitidos": PERMITIR_PLUGINS,
    }


@router.get("/optimizadores/{nombre}")
async def obtener(nombre: str, request: Request) -> dict[str, Any]:
    """Descriptor con los hiperparámetros deducidos de la firma del constructor."""
    try:
        return _servicio(request).obtener(nombre).a_dict()
    except OptimizadorDesconocido as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/optimizadores/validar")
async def validar(dto: ValidacionOptimizadorDTO, request: Request) -> dict[str, Any]:
    """Comprueba los hiperparámetros contra la firma real, antes de gastar CPU."""
    try:
        return _servicio(request).validar(dto.nombre, dto.params)
    except OptimizadorDesconocido as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/optimizadores/recargar-plugins", status_code=status.HTTP_200_OK)
async def recargar_plugins(request: Request) -> dict[str, Any]:
    """Re-escanea `web_interface/plugins/`.

    No recibe código: el usuario deposita el archivo con su editor y esto solo
    vuelve a leer el directorio.
    """
    if not PERMITIR_PLUGINS:
        raise HTTPException(
            status_code=403,
            detail="La carga de plugins está desactivada en la configuración",
        )
    return _servicio(request).recargar_plugins()


@router.get("/funciones-perdida")
async def funciones_perdida(request: Request) -> list[str]:
    return request.app.state.registro_perdidas.listar()

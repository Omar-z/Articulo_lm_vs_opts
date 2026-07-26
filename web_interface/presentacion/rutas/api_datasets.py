"""API de datasets: catálogo, vista previa, estadísticas, esquema y alta."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from web_interface.infraestructura.repositorio_datasets import (
    DatasetNoEncontrado,
    DatasetRechazado,
)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _servicio(request: Request):
    return request.app.state.servicio_datasets


def _repo(request: Request):
    return request.app.state.repositorio_datasets


def _opciones(sep: str | None, header: str | None) -> dict[str, Any]:
    """Traduce los parámetros de consulta a opciones del cargador."""
    opciones: dict[str, Any] = {}
    if sep is not None:
        #`\t` llega escapado desde la URL.
        opciones["sep"] = "\t" if sep in ("\\t", "tab") else sep
    if header is not None:
        if header in ("", "null", "none", "ninguno"):
            opciones["header"] = None
        else:
            try:
                opciones["header"] = int(header)
            except ValueError:
                opciones["header"] = None
    return opciones


@router.get("")
async def listar(request: Request, origen: str | None = None) -> list[dict[str, Any]]:
    return [d.a_dict() for d in _repo(request).listar(origen)]


@router.get("/{id_dataset}/vista-previa")
async def vista_previa(
    id_dataset: str,
    request: Request,
    filas: int = 20,
    sep: str | None = None,
    header: str | None = None,
) -> dict[str, Any]:
    try:
        return _servicio(request).vista_previa(
            id_dataset, _opciones(sep, header), min(filas, 200)
        )
    except DatasetNoEncontrado as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - error de lectura del archivo
        raise HTTPException(
            status_code=422, detail=f"No se pudo leer el dataset: {exc}"
        ) from exc


@router.get("/{id_dataset}/estadisticas")
async def estadisticas(
    id_dataset: str, request: Request, sep: str | None = None, header: str | None = None
) -> dict[str, Any]:
    try:
        return _servicio(request).estadisticas(id_dataset, _opciones(sep, header))
    except DatasetNoEncontrado as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"No se pudieron calcular las estadísticas: {exc}"
        ) from exc


@router.get("/{id_dataset}/esquema-sugerido")
async def esquema_sugerido(
    id_dataset: str, request: Request, sep: str | None = None, header: str | None = None
) -> dict[str, Any]:
    """Propuesta de configuración deducida del contenido del archivo."""
    try:
        return _servicio(request).esquema_sugerido(id_dataset, _opciones(sep, header))
    except DatasetNoEncontrado as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"No se pudo inferir el esquema: {exc}"
        ) from exc


@router.post("", status_code=status.HTTP_201_CREATED)
async def subir(request: Request, archivo: UploadFile = File(...)) -> dict[str, Any]:
    """Da de alta un archivo de datos.

    Solo se aceptan extensiones de datos y nunca se importan como código: se leen
    con pandas o scipy.
    """
    try:
        dataset = _repo(request).dar_de_alta(archivo.filename or "dataset", archivo.file)
    except DatasetRechazado as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await archivo.close()
    return dataset.a_dict()


@router.delete("/{id_dataset}", status_code=status.HTTP_204_NO_CONTENT)
async def eliminar(id_dataset: str, request: Request) -> None:
    try:
        _repo(request).eliminar(id_dataset)
    except DatasetNoEncontrado as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DatasetRechazado as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

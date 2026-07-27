"""Vistas HTML y endpoints de servicio general."""

from __future__ import annotations

import sys

import torch
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from web_interface import __version__
from web_interface.configuracion import RAIZ_PROYECTO
from web_interface.infraestructura.dispositivos import listar_dispositivos
from web_interface.presentacion.plantillas import plantillas

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def inicio(request: Request) -> HTMLResponse:
    """Portada: accesos a las secciones y estado del laboratorio."""
    return plantillas.TemplateResponse(
        request,
        "index.html",
        {"titulo": "Laboratorio ANFIS", "seccion": "inicio"},
    )


@router.get("/trabajos", response_class=HTMLResponse)
async def trabajos(request: Request) -> HTMLResponse:
    """Cola de ejecución e historial de trabajos de la sesión."""
    return plantillas.TemplateResponse(
        request,
        "trabajos.html",
        {"titulo": "Trabajos", "seccion": "trabajos"},
    )


@router.get("/monitor/{id_trabajo}", response_class=HTMLResponse)
async def monitor(id_trabajo: str, request: Request) -> HTMLResponse:
    """Monitor en vivo de un trabajo: curva de pérdida, progreso y bitácora."""
    return plantillas.TemplateResponse(
        request,
        "monitor.html",
        {
            "titulo": f"Monitor · {id_trabajo}",
            "seccion": "trabajos",
            "id_trabajo": id_trabajo,
        },
    )


@router.get("/historial", response_class=HTMLResponse)
async def historial(request: Request) -> HTMLResponse:
    """Experimentos terminados: los de la web y los que dejó el script."""
    return plantillas.TemplateResponse(
        request,
        "historial.html",
        {"titulo": "Historial", "seccion": "historial"},
    )


@router.get("/historial/{origen}/{id_experimento}", response_class=HTMLResponse)
async def resultado_detalle(
    origen: str, id_experimento: str, request: Request
) -> HTMLResponse:
    """Tablero comparativo de un experimento."""
    return plantillas.TemplateResponse(
        request,
        "resultado_detalle.html",
        {
            "titulo": f"Resultado · {id_experimento}",
            "seccion": "historial",
            "origen": origen,
            "id_experimento": id_experimento,
        },
    )


@router.get("/datasets", response_class=HTMLResponse)
async def datasets(request: Request) -> HTMLResponse:
    """Explorador de datasets: los de `data_sets/` y los subidos."""
    return plantillas.TemplateResponse(
        request,
        "datasets.html",
        {"titulo": "Datasets", "seccion": "datasets"},
    )


@router.get("/datasets/{id_dataset}", response_class=HTMLResponse)
async def dataset_detalle(id_dataset: str, request: Request) -> HTMLResponse:
    """Vista previa y estadísticas por columna de un dataset."""
    return plantillas.TemplateResponse(
        request,
        "dataset_detalle.html",
        {"titulo": "Dataset", "seccion": "datasets", "id_dataset": id_dataset},
    )


@router.get("/optimizadores", response_class=HTMLResponse)
async def optimizadores(request: Request) -> HTMLResponse:
    """Catálogo de optimizadores y gestión de plugins."""
    return plantillas.TemplateResponse(
        request,
        "optimizadores.html",
        {"titulo": "Optimizadores", "seccion": "optimizadores"},
    )


@router.get("/constructor", response_class=HTMLResponse)
async def constructor(request: Request) -> HTMLResponse:
    """Constructor visual de experimentos."""
    return plantillas.TemplateResponse(
        request,
        "constructor.html",
        {"titulo": "Constructor de experimentos", "seccion": "constructor"},
    )


@router.get("/politicas", response_class=HTMLResponse)
async def politicas(request: Request) -> HTMLResponse:
    """Catálogo y editor de políticas de paro."""
    return plantillas.TemplateResponse(
        request,
        "politicas.html",
        {"titulo": "Políticas de paro", "seccion": "politicas"},
    )


@router.get("/hiperparametros", response_class=HTMLResponse)
async def hiperparametros(request: Request) -> HTMLResponse:
    """Barrido de learning rate por optimizador y número de reglas.

    Ya no está en la navegación —su sitio lo ocupa Políticas—, pero sigue
    accesible por URL y el barrido funciona.
    """
    return plantillas.TemplateResponse(
        request,
        "hiperparametros.html",
        {"titulo": "Hiperparámetros", "seccion": "hiperparametros"},
    )


@router.get("/api/salud")
async def salud() -> dict[str, object]:
    """Diagnóstico de arranque: raíz detectada, versiones y dispositivos."""
    return {
        "ok": True,
        "version": __version__,
        "raiz": str(RAIZ_PROYECTO),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "dispositivos": [d.a_dict() for d in listar_dispositivos()],
    }


@router.get("/api/dispositivos")
async def dispositivos() -> list[dict[str, object]]:
    """Dispositivos de cómputo con su compatibilidad con el modelo ANFIS."""
    return [d.a_dict() for d in listar_dispositivos()]

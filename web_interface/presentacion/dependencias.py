"""Contenedor de dependencias de la aplicación.

Los componentes de larga vida (registros, gestor de trabajos, repositorios) se
guardan en ``app.state`` durante el arranque y se obtienen desde los endpoints con
``Depends``.

Deliberadamente **no** se usan singletons de módulo: el núcleo ya sufre ese
antipatrón (`rpipeline.dispositivo`, `graficar`, `minilotes`, `mejores_parametros`
y `mp_path` son globales mutados desde ``__main__``) y no se repite aquí.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request

from web_interface.presentacion.plantillas import plantillas as _plantillas


def obtener_estado(request: Request) -> Any:
    """Acceso crudo a ``app.state``."""
    return request.app.state


def obtener_plantillas() -> Any:
    """Motor Jinja2 compartido."""
    return _plantillas

"""Construcción de la aplicación FastAPI.

`configuracion` se importa primero a propósito: fija ``MPLBACKEND`` y añade la raíz
del proyecto a ``sys.path`` antes de que nada toque el núcleo.

Los componentes de larga vida se crean en el ciclo de vida y se guardan en
``app.state``, de donde los toman los endpoints. No hay singletons de módulo: el
núcleo ya demuestra el daño de ese antipatrón con `rpipeline.dispositivo`,
`graficar`, `minilotes`, `mejores_parametros` y `mp_path`.
"""

from __future__ import annotations

import web_interface.configuracion as configuracion  # noqa: F401  (efectos de import)

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from web_interface import __version__
from web_interface.aplicacion.ensambladores import EnsambladorDeConfig
from web_interface.aplicacion.fachada_anfis import FachadaANFIS
from web_interface.aplicacion.gestor_trabajos import GestorDeTrabajos
from web_interface.aplicacion.servicio_datasets import ServicioDeDatasets
from web_interface.aplicacion.servicio_optimizadores import ServicioDeOptimizadores
from web_interface.aplicacion.servicio_politicas import ServicioDePoliticas
from web_interface.configuracion import (
    DIR_ESTATICOS,
    DIR_HIPERPARAMETROS,
    asegurar_directorios,
)
from web_interface.infraestructura.cargadores import FabricaDeCargadores
from web_interface.infraestructura.editor_fis import EditorDeFIS
from web_interface.infraestructura.repositorio_datasets import RepositorioDeDatasets
from web_interface.infraestructura.repositorio_plugins import RepositorioDePlugins
from web_interface.infraestructura.repositorio_politicas import RepositorioDePoliticas
from web_interface.infraestructura.registros import (
    crear_registro_optimizadores,
    crear_registro_perdidas,
)
from web_interface.infraestructura.repositorio_experimentos import (
    RepositorioDeExperimentos,
)
from web_interface.infraestructura.repositorio_resultados import (
    FuenteRepoLegado,
    FuenteWeb,
    RepositorioDeResultados,
)
from web_interface.infraestructura.salida import (
    desinstalar_multiplexor,
    instalar_multiplexor,
)
from web_interface.presentacion.esquemas import ConfigDTO
from web_interface.presentacion.puente_ws import PuenteDeEventos
from web_interface.presentacion.rutas import (
    api_datasets,
    api_experimentos,
    api_optimizadores,
    api_politicas,
    api_resultados,
    api_trabajos,
    paginas,
    ws_progreso,
)


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Arranque y apagado ordenados de los componentes de larga vida."""
    asegurar_directorios()

    #Registros y colaboradores compartidos
    app.state.registro_optimizadores = crear_registro_optimizadores()
    app.state.registro_perdidas = crear_registro_perdidas()
    app.state.fabrica_cargadores = FabricaDeCargadores()
    app.state.editor_fis = EditorDeFIS()
    app.state.ensamblador = EnsambladorDeConfig(
        app.state.registro_optimizadores, app.state.registro_perdidas
    )

    def crear_fachada(
        dispositivo: str = "cpu",
        minilotes: bool = False,
        forzar_float64: bool = False,
    ) -> FachadaANFIS:
        """Una fachada por trabajo: el dispositivo y el modo de lotes son suyos."""
        return FachadaANFIS(
            registro_optimizadores=app.state.registro_optimizadores,
            registro_perdidas=app.state.registro_perdidas,
            fabrica_cargadores=app.state.fabrica_cargadores,
            editor_fis=app.state.editor_fis,
            dispositivo=dispositivo,
            minilotes=minilotes,
            forzar_float64=forzar_float64,
        )

    app.state.crear_fachada = crear_fachada

    #Catálogo de optimizadores: PyTorch + LM + los plugins del usuario
    app.state.servicio_optimizadores = ServicioDeOptimizadores(
        app.state.registro_optimizadores, RepositorioDePlugins()
    )
    app.state.servicio_optimizadores.recargar_plugins()

    #Políticas de paro: las del núcleo más las que escriba el usuario. Solo se
    #pueden editar mientras el servidor sea local (ver repositorio_politicas).
    app.state.servicio_politicas = ServicioDePoliticas(
        RepositorioDePoliticas(permitir_escritura=not configuracion.EXPUESTO_EN_RED)
    )

    #Repositorios
    app.state.repositorio_datasets = RepositorioDeDatasets()
    app.state.servicio_datasets = ServicioDeDatasets(
        app.state.repositorio_datasets, app.state.fabrica_cargadores
    )
    app.state.repositorio_experimentos = RepositorioDeExperimentos()
    app.state.fuente_web = FuenteWeb()
    app.state.repositorio_resultados = RepositorioDeResultados(
        [app.state.fuente_web, FuenteRepoLegado()]
    )

    #Puente hilo -> event loop para los WebSockets
    app.state.puente = PuenteDeEventos()
    app.state.puente.enlazar(asyncio.get_running_loop())

    #Multiplexor de stdout: aísla la barra ANSI del núcleo de la consola de uvicorn
    app.state.multiplexor = instalar_multiplexor()

    def guardar_resultado(trabajo, resultado) -> str | None:
        """Persiste el barrido al terminar, aunque se haya cancelado.

        Un trabajo de horas no debe perder lo ya calculado, así que también se
        guardan los resultados parciales.
        """
        dto = ConfigDTO(**trabajo.metadatos["config"])

        if resultado.get("tipo") == "hiperparametros":
            #Se escribe con el mismo esquema que produce `mejores_parametros.py`,
            #para poder usarlo tal cual con el CLI.
            destino = DIR_HIPERPARAMETROS / f"{trabajo.id}.json"
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text(
                json.dumps(resultado["parametros"], indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
            (destino.parent / f"{trabajo.id}_detalle.json").write_text(
                json.dumps(resultado["detalle"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return str(destino)

        carpeta = app.state.fuente_web.guardar(
            trabajo.id,
            app.state.ensamblador.desde_dto(dto),
            resultado,
            config_cruda=app.state.ensamblador.hacia_json_cli(dto),
        )
        return str(carpeta)

    app.state.gestor = GestorDeTrabajos(
        multiplexor=app.state.multiplexor,
        observadores_globales=[app.state.puente],
        al_terminar=guardar_resultado,
    )
    app.state.gestor.iniciar()

    try:
        yield
    finally:
        app.state.gestor.detener()
        app.state.puente.desenlazar()
        desinstalar_multiplexor()


def crear_app() -> FastAPI:
    """Ensambla la aplicación: rutas, estáticos y ciclo de vida."""
    app = FastAPI(
        title="Laboratorio ANFIS",
        description=(
            "Interfaz web para configurar, ejecutar y comparar experimentos "
            "del modelo neurodifuso con distintos optimizadores."
        ),
        version=__version__,
        lifespan=ciclo_de_vida,
    )

    app.mount(
        "/estaticos",
        StaticFiles(directory=str(DIR_ESTATICOS)),
        name="estaticos",
    )

    app.include_router(paginas.router)
    app.include_router(api_optimizadores.router)
    app.include_router(api_politicas.router)
    app.include_router(api_datasets.router)
    app.include_router(api_trabajos.router)
    app.include_router(api_experimentos.router)
    app.include_router(api_resultados.router)
    app.include_router(ws_progreso.router)

    return app


app = crear_app()

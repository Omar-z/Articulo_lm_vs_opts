"""Servicio del catálogo de optimizadores.

Coordina el registro con el repositorio de plugins, de modo que dar de alta un
optimizador nuevo sea dejar un archivo en `plugins/` y pulsar «Recargar».
"""

from __future__ import annotations

from typing import Any

from web_interface.dominio.modelos import DescriptorOptimizador
from web_interface.infraestructura.registros import RegistroDeOptimizadores
from web_interface.infraestructura.repositorio_plugins import (
    ErrorDePlugin,
    RepositorioDePlugins,
)


class ServicioDeOptimizadores:
    """Consulta y recarga del catálogo."""

    def __init__(
        self,
        registro: RegistroDeOptimizadores,
        repositorio_plugins: RepositorioDePlugins,
    ) -> None:
        self.registro = registro
        self.repositorio_plugins = repositorio_plugins
        self.ultimos_errores: list[ErrorDePlugin] = []

    def recargar_plugins(self) -> dict[str, Any]:
        """Re-escanea `plugins/` y actualiza el registro.

        Se vacían primero los plugins ya registrados para que borrar un archivo
        también lo quite del catálogo.
        """
        self.registro.vaciar_plugins()
        encontrados, errores = self.repositorio_plugins.descubrir()
        self.ultimos_errores = errores

        cargados: list[str] = []
        for nombre, clase, modulo in encontrados:
            #Un plugin no puede desplazar a un optimizador de PyTorch o al LM.
            nombre_final = nombre
            if self.registro.existe(nombre) and self.registro.obtener(nombre).origen != "plugin":
                nombre_final = f"{nombre} (plugin)"
            self.registro.registrar_plugin(nombre_final, clase, modulo)
            cargados.append(nombre_final)

        return {
            "cargados": cargados,
            "errores": [e.a_dict() for e in errores],
            "permitido": self.repositorio_plugins.permitir,
            "directorio": str(self.repositorio_plugins.directorio),
        }

    def listar(self, origen: str | None = None) -> list[DescriptorOptimizador]:
        return self.registro.listar(origen)

    def obtener(self, nombre: str) -> DescriptorOptimizador:
        return self.registro.obtener(nombre)

    def validar(self, nombre: str, params: dict[str, Any]) -> dict[str, Any]:
        normalizados, errores = self.registro.validar_parametros(nombre, params)
        return {
            "valido": not errores,
            "errores": errores,
            #Las tuplas se pasan a lista para que el JSON sea válido.
            "params_normalizados": {
                clave: list(valor) if isinstance(valor, tuple) else valor
                for clave, valor in normalizados.items()
            },
        }

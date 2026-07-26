"""Carga de optimizadores propios desde `web_interface/plugins/`.

Permite añadir optimizadores sin tocar el código de la aplicación: basta dejar un
`.py` en esa carpeta y pulsar «Recargar» en el catálogo.

**Aviso de seguridad.** Importar un módulo es ejecutar su código con los permisos
de este proceso. Por eso:

* El código **no se sube por la web**: el endpoint solo re-escanea el directorio;
  el usuario deposita el archivo con su editor. La subida por HTTP está limitada a
  archivos de datos, que jamás se importan.
* El servidor escucha en `127.0.0.1` salvo que se pida lo contrario de forma
  explícita.
* `configuracion.PERMITIR_PLUGINS` desactiva el escaneo por completo.
* Cada archivo se carga aislado: uno roto devuelve un error legible y no tumba el
  servidor.

No se intenta un aislamiento real (subproceso restringido, lista blanca de AST):
daría una falsa sensación de seguridad y rompería optimizadores legítimos que
importan numpy o torch.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch

from web_interface.configuracion import DIR_PLUGINS, PERMITIR_PLUGINS


@dataclass(slots=True)
class ErrorDePlugin:
    """Un archivo de plugins que no se pudo cargar."""

    archivo: str
    mensaje: str

    def a_dict(self) -> dict[str, str]:
        return {"archivo": self.archivo, "mensaje": self.mensaje}


class RepositorioDePlugins:
    """Descubre clases de optimizador en los archivos de `plugins/`."""

    def __init__(
        self, directorio: Path | None = None, permitir: bool | None = None
    ) -> None:
        self.directorio = directorio or DIR_PLUGINS
        self.permitir = PERMITIR_PLUGINS if permitir is None else permitir

    def descubrir(self) -> tuple[list[tuple[str, type, str]], list[ErrorDePlugin]]:
        """Devuelve (nombre, clase, módulo) de cada optimizador válido, y los errores."""
        if not self.permitir or not self.directorio.exists():
            return [], []

        encontrados: list[tuple[str, type, str]] = []
        errores: list[ErrorDePlugin] = []

        for archivo in sorted(self.directorio.glob("*.py")):
            if archivo.name.startswith("_"):
                continue
            try:
                modulo = self._cargar_modulo(archivo)
            except Exception as exc:  # noqa: BLE001 - un plugin roto no tumba el servidor
                errores.append(
                    ErrorDePlugin(archivo.name, f"{type(exc).__name__}: {exc}")
                )
                continue

            for nombre, clase in self._clases_validas(modulo):
                encontrados.append((nombre, clase, archivo.name))

        return encontrados, errores

    # -- internos ------------------------------------------------------------

    @staticmethod
    def _cargar_modulo(archivo: Path) -> ModuleType:
        nombre_modulo = f"web_interface_plugins.{archivo.stem}"
        especificacion = importlib.util.spec_from_file_location(nombre_modulo, archivo)
        if especificacion is None or especificacion.loader is None:
            raise ImportError(f"No se pudo preparar la carga de '{archivo.name}'")

        modulo = importlib.util.module_from_spec(especificacion)
        #Se registra antes de ejecutar para que funcionen las referencias internas.
        sys.modules[nombre_modulo] = modulo
        try:
            especificacion.loader.exec_module(modulo)
        except Exception:
            sys.modules.pop(nombre_modulo, None)
            raise
        return modulo

    @staticmethod
    def _clases_validas(modulo: ModuleType) -> list[tuple[str, type]]:
        """Filtra las clases del módulo que sirven como optimizador.

        Los criterios, en orden:

        1. Deriva de `torch.optim.Optimizer` o del `Optimizador` del núcleo.
        2. Está **definida** en este módulo, no solo importada — de lo contrario un
           plugin que hiciera ``from torch.optim import Adam`` volvería a registrar
           Adam.
        3. Su constructor es introspeccionable, porque de ahí sale el formulario.
        4. No se llama `nombre = "LM"`: ese atributo es el discriminante con que
           `train_nfs` decide cómo invocar `step()` (`V2_Anfis.py:437`), y un choque
           haría que un optimizador de PyTorch recibiera `step(X, y)`.
        """
        from web_interface.nucleo import Optimizador as OptimizadorNucleo

        validas: list[tuple[str, type]] = []
        for nombre, objeto in inspect.getmembers(modulo, inspect.isclass):
            if objeto in (torch.optim.Optimizer, OptimizadorNucleo):
                continue
            if not (
                issubclass(objeto, torch.optim.Optimizer)
                or issubclass(objeto, OptimizadorNucleo)
            ):
                continue
            if getattr(objeto, "__module__", "") != modulo.__name__:
                continue
            if getattr(objeto, "nombre", None) == "LM":
                continue
            try:
                inspect.signature(objeto.__init__)
            except (ValueError, TypeError):
                continue
            validas.append((nombre, objeto))
        return validas

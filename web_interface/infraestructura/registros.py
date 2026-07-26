"""Registros de optimizadores y funciones de pérdida — patrón Registry.

El núcleo tiene estos catálogos como diccionarios literales duplicados en
`rpipeline.py:74-89` y `mejores_parametros.py:12-27`. Un diccionario cerrado no
admite dar de alta optimizadores nuevos sin editar código, que es justo lo que se
pide aquí, así que se sustituyen por registros con altas dinámicas.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import torch

from web_interface.dominio.modelos import DescriptorOptimizador
from web_interface.infraestructura.introspeccion import IntrospectorDeOptimizadores


class OptimizadorDesconocido(KeyError):
    """Se pidió un optimizador que no está registrado."""


class FuncionPerdidaDesconocida(KeyError):
    """Se pidió una función de pérdida que no está registrada."""


class RegistroDeOptimizadores:
    """Catálogo de optimizadores disponibles: los de PyTorch, el LM propio y los plugins."""

    def __init__(self, introspector: IntrospectorDeOptimizadores | None = None) -> None:
        self._introspector = introspector or IntrospectorDeOptimizadores()
        self._descriptores: dict[str, DescriptorOptimizador] = {}

    # -- altas ---------------------------------------------------------------

    def registrar_torch(self) -> int:
        """Da de alta todos los `torch.optim.Optimizer` disponibles. Devuelve cuántos."""
        for nombre, clase in self._introspector.descubrir_torch():
            self._descriptores[nombre] = self._introspector.describir(
                clase, nombre, origen="torch"
            )
        return sum(1 for d in self._descriptores.values() if d.origen == "torch")

    def registrar_propio(self, nombre: str, clase: type) -> DescriptorOptimizador:
        """Da de alta un optimizador del núcleo (el LM), que no deriva de torch.optim."""
        descriptor = self._introspector.describir(clase, nombre, origen="propio")
        self._descriptores[nombre] = descriptor
        return descriptor

    def registrar_plugin(
        self, nombre: str, clase: type, modulo: str
    ) -> DescriptorOptimizador:
        """Da de alta una clase encontrada en `web_interface/plugins/`."""
        descriptor = self._introspector.describir(clase, nombre, origen="plugin")
        descriptor.modulo = modulo
        self._descriptores[nombre] = descriptor
        return descriptor

    def vaciar_plugins(self) -> None:
        """Elimina los plugins registrados, antes de un re-escaneo del directorio."""
        for nombre in [
            n for n, d in self._descriptores.items() if d.origen == "plugin"
        ]:
            del self._descriptores[nombre]

    # -- consultas -----------------------------------------------------------

    def obtener(self, nombre: str) -> DescriptorOptimizador:
        descriptor = self._descriptores.get(nombre)
        if descriptor is None:
            raise OptimizadorDesconocido(
                f"El optimizador '{nombre}' no está registrado. "
                f"Disponibles: {', '.join(sorted(self._descriptores))}"
            )
        return descriptor

    def existe(self, nombre: str) -> bool:
        return nombre in self._descriptores

    def listar(self, origen: str | None = None) -> list[DescriptorOptimizador]:
        descriptores = list(self._descriptores.values())
        if origen is not None:
            descriptores = [d for d in descriptores if d.origen == origen]
        #Primero los compatibles, luego por origen y nombre.
        return sorted(
            descriptores,
            key=lambda d: (not d.compatible, d.origen != "propio", d.nombre.lower()),
        )

    # -- validación ----------------------------------------------------------

    def validar_parametros(
        self, nombre: str, params: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """Comprueba que `params` encaja con la firma real del constructor.

        Se hace aquí y no al construir el modelo porque un error de nombre solo se
        vería tras minutos de barrido, cuando ya se ha gastado CPU.

        @return: (parámetros normalizados, lista de errores)
        """
        descriptor = self.obtener(nombre)
        errores: list[str] = []
        normalizados: dict[str, Any] = {}

        por_nombre = {h.nombre: h for h in descriptor.hiperparametros}
        for clave, valor in params.items():
            if clave in self._introspector.INYECTADOS:
                #`device` lo inyecta la fachada; se ignora si viene del formulario.
                continue
            hiper = por_nombre.get(clave)
            if hiper is None:
                errores.append(
                    f"'{clave}' no es un parámetro de {nombre}. "
                    f"Acepta: {', '.join(sorted(por_nombre)) or '(ninguno)'}"
                )
                continue
            try:
                normalizados[clave] = _convertir(valor, hiper.tipo, hiper.aridad)
            except (TypeError, ValueError) as exc:
                errores.append(f"'{clave}': {exc}")

        #Comprobación final contra la firma real.
        if not errores:
            try:
                firma = inspect.signature(descriptor.clase.__init__)
                firma.bind_partial(None, None, **normalizados)
            except TypeError as exc:
                errores.append(str(exc))

        return normalizados, errores


def _convertir(valor: Any, tipo: str, aridad: int) -> Any:
    """Lleva un valor del formulario al tipo que espera el constructor."""
    if valor is None:
        return None

    if tipo == "bool":
        if isinstance(valor, bool):
            return valor
        if isinstance(valor, str):
            return valor.lower() in ("true", "1", "si", "sí", "on")
        return bool(valor)

    if tipo == "bool_opcional":
        if valor in (None, "", "auto"):
            return None
        return _convertir(valor, "bool", 1)

    if tipo == "tupla":
        if not isinstance(valor, (list, tuple)):
            raise ValueError(f"se esperaba una lista de {aridad} valores")
        if len(valor) != aridad:
            raise ValueError(f"se esperaban {aridad} valores, llegaron {len(valor)}")
        return tuple(None if v is None else float(v) for v in valor)

    if tipo == "int":
        return int(valor)

    if tipo == "float":
        return float(valor)

    return valor


class RegistroDeFuncionesPerdida:
    """Catálogo de funciones de pérdida.

    A diferencia del `FN_LOSS` del núcleo, que guarda **instancias** compartidas
    entre todos los modelos, aquí se guardan fábricas y se devuelve una instancia
    nueva por modelo. Con `MSELoss` da igual, pero evita compartir estado si alguna
    vez se registra una pérdida que lo tenga.
    """

    def __init__(self) -> None:
        self._fabricas: dict[str, Callable[[], Callable]] = {}

    def registrar(self, nombre: str, fabrica: Callable[[], Callable]) -> None:
        self._fabricas[nombre] = fabrica

    def obtener(self, nombre: str) -> Callable:
        fabrica = self._fabricas.get(nombre)
        if fabrica is None:
            raise FuncionPerdidaDesconocida(
                f"La función de pérdida '{nombre}' no está registrada. "
                f"Disponibles: {', '.join(sorted(self._fabricas))}"
            )
        return fabrica()

    def existe(self, nombre: str) -> bool:
        return nombre in self._fabricas

    def listar(self) -> list[str]:
        return sorted(self._fabricas)


# --- Fábricas por defecto, equivalentes a FN_LOSS de rpipeline.py:83-89 ----------

def _sse(salida: torch.Tensor, objetivo: torch.Tensor) -> torch.Tensor:
    return torch.sum((salida - objetivo) ** 2)


def _rmse(salida: torch.Tensor, objetivo: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((salida - objetivo) ** 2))


def crear_registro_perdidas() -> RegistroDeFuncionesPerdida:
    """Registro con las mismas pérdidas que usa el CLI."""
    registro = RegistroDeFuncionesPerdida()
    registro.registrar("MSE", torch.nn.MSELoss)
    registro.registrar("RMSE", lambda: _rmse)
    registro.registrar("MAE", torch.nn.L1Loss)
    registro.registrar("SSE", lambda: _sse)
    registro.registrar("Entropia", torch.nn.CrossEntropyLoss)
    return registro


#Métricas que se calculan siempre, además de la pérdida elegida
#(equivalente a `metricas_importantes` en rpipeline.py:92-94).
NOMBRES_METRICAS: tuple[str, ...] = ("RMSE", "MSE", "MAE", "SSE")


def crear_registro_optimizadores() -> RegistroDeOptimizadores:
    """Registro poblado con los optimizadores de PyTorch y el LM del proyecto."""
    from web_interface.nucleo import LevenberMaquardtOpt

    registro = RegistroDeOptimizadores()
    registro.registrar_torch()
    registro.registrar_propio("LM", LevenberMaquardtOpt)
    return registro

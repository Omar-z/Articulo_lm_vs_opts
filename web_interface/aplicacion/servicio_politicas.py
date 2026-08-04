"""Catálogo de políticas de paro: las del núcleo y las que escribe el usuario.

Reúne en una sola lista las cuatro políticas de `funciones_auxiliares` y las que
el usuario define en `politicas/politicas_usuario.py`, y construye los prototipos
que cada barrido necesita.
"""

from __future__ import annotations

import inspect
from typing import Any

from web_interface.infraestructura.repositorio_politicas import (
    PoliticaDisponible,
    RepositorioDePoliticas,
)
from web_interface.nucleo import (
    DataConfig,
    PoliticaDeParo,
    PoliticaFallos,
    PoliticaLambdaLM,
    PoliticaNanOrInf,
    PoliticaTolerancia,
)

#Políticas del núcleo, en el mismo orden que usa `rpipeline.py:196-206`.
#NaN/Inf va primero para cortar en cuanto una corrida diverge, antes de gastar el
#resto de las épocas.
POLITICAS_NUCLEO: tuple[tuple[str, type, str], ...] = (
    (
        "PoliticaNanOrInf",
        PoliticaNanOrInf,
        "Detiene si la pérdida se vuelve NaN o infinita: la corrida ha divergido",
    ),
    (
        "PoliticaTolerancia",
        PoliticaTolerancia,
        "Detiene cuando la pérdida baja del umbral de tolerancia del experimento",
    ),
    (
        "PoliticaFallos",
        PoliticaFallos,
        "Contador multiplicativo: se dispara tras varias épocas seguidas sin mejorar",
    ),
    (
        "PoliticaLambdaLM",
        PoliticaLambdaLM,
        "Solo para el Levenberg-Marquardt: detiene si lambda supera su máximo",
    ),
)

NOMBRES_NUCLEO: tuple[str, ...] = tuple(n for n, _, _ in POLITICAS_NUCLEO)


class ServicioDePoliticas:
    """Consulta del catálogo y construcción de los prototipos de un barrido."""

    def __init__(self, repositorio: RepositorioDePoliticas) -> None:
        self.repositorio = repositorio
        self.politicas_usuario: list[PoliticaDisponible] = []
        self.error_de_carga: str | None = None
        self.recargar()

    # -- catálogo ------------------------------------------------------------

    def recargar(self) -> dict[str, Any]:
        """Vuelve a leer el archivo de políticas del usuario."""
        self.politicas_usuario, self.error_de_carga = self.repositorio.cargar()
        return {
            "cargadas": [p.nombre_clase for p in self.politicas_usuario],
            "error": self.error_de_carga,
            "archivo": str(self.repositorio.archivo),
            "escritura_permitida": self.repositorio.permitir_escritura,
        }

    def listar(self) -> list[PoliticaDisponible]:
        """Todas las políticas disponibles, las del núcleo primero."""
        del_nucleo = [
            PoliticaDisponible(
                nombre_clase=nombre,
                etiqueta=_etiqueta_del_nucleo(clase, nombre),
                origen="nucleo",
                doc=doc,
                editable=False,
                codigo="",
                parametros=_parametros(clase),
            )
            for nombre, clase, doc in POLITICAS_NUCLEO
        ]
        return del_nucleo + list(self.politicas_usuario)

    def existe(self, nombre_clase: str) -> bool:
        return any(p.nombre_clase == nombre_clase for p in self.listar())

    # -- prototipos ----------------------------------------------------------

    def prototipos(
        self, config: DataConfig, seleccion: list[str] | None = None
    ) -> list[PoliticaDeParo]:
        """Instancia las políticas elegidas para un barrido.

        Sin selección se usan las cuatro del núcleo, que es el comportamiento del
        CLI. El orden de la selección se respeta: `CompositorDePoliticas` corta en
        cuanto una devuelve True, así que quien va antes se lleva la atribución
        del motivo de paro.
        """
        nombres = list(seleccion) if seleccion else list(NOMBRES_NUCLEO)
        exp = config.experimentos

        #`PoliticaFallos` toma sus parámetros del LM si está en el experimento,
        #igual que hace `rpipeline.py:199-204`.
        lm_params: dict[str, Any] | None = None
        for opt in config.optimizadores:
            if opt.nombre == "LM":
                lm_params = opt.params
                break

        instancias: list[PoliticaDeParo] = []
        for nombre in nombres:
            instancia = self._instanciar(nombre, exp, lm_params)
            if instancia is not None:
                instancias.append(instancia)
        return instancias

    def _instanciar(
        self, nombre: str, exp: Any, lm_params: dict[str, Any] | None
    ) -> PoliticaDeParo | None:
        if nombre == "PoliticaNanOrInf":
            return PoliticaNanOrInf()
        if nombre == "PoliticaTolerancia":
            return PoliticaTolerancia(exp.tolerancia)
        if nombre == "PoliticaFallos":
            return PoliticaFallos(
                limite=1e10,
                init=lm_params.get("lambda_init", 0.01) if lm_params else 0.01,
                inc=lm_params.get("lambda_incr", 10) if lm_params else 10,
                dec=lm_params.get("lambda_decr", 0.1) if lm_params else 0.1,
            )
        if nombre == "PoliticaLambdaLM":
            return PoliticaLambdaLM()

        #Política del usuario: se construye sin argumentos, por eso se exige que
        #todos los parámetros de `__init__` tengan valor por defecto.
        try:
            return self.repositorio.clase(nombre)()
        except Exception:  # noqa: BLE001 - una política rota no debe impedir el barrido
            return None

    # -- edición -------------------------------------------------------------

    def validar(self, nombre_clase: str, codigo: str) -> list[str]:
        return self.repositorio.validar(nombre_clase, codigo)

    def probar(self, nombre_clase: str, codigo: str) -> dict[str, Any]:
        return self.repositorio.probar(nombre_clase, codigo)

    def guardar(self, nombre_clase: str, codigo: str) -> PoliticaDisponible:
        politica = self.repositorio.guardar(nombre_clase, codigo)
        self.recargar()
        return politica

    def eliminar(self, nombre_clase: str) -> None:
        if nombre_clase in NOMBRES_NUCLEO:
            raise PermissionError(
                f"'{nombre_clase}' es del núcleo del proyecto y no se puede borrar "
                "desde aquí"
            )
        self.repositorio.eliminar(nombre_clase)
        self.recargar()


def _etiqueta_del_nucleo(clase: type, por_defecto: str) -> str:
    """Nombre legible de una política del núcleo: el que fija en su `__init__`."""
    try:
        return str(getattr(clase(), "nombre", por_defecto))
    except Exception:  # noqa: BLE001
        return por_defecto


def _parametros(clase: type) -> list[dict[str, Any]]:
    try:
        firma = inspect.signature(clase.__init__)
    except (ValueError, TypeError):
        return []
    salida = []
    for indice, (nombre, parametro) in enumerate(firma.parameters.items()):
        if indice == 0 or nombre == "self":
            continue
        salida.append(
            {
                "nombre": nombre,
                "por_defecto": (
                    None
                    if parametro.default is inspect.Parameter.empty
                    else str(parametro.default)
                ),
            }
        )
    return salida

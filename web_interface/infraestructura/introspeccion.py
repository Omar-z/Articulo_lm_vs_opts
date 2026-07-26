"""Descripción automática de los optimizadores por introspección.

En vez de mantener a mano una tabla de hiperparámetros por optimizador (que se
quedaría obsoleta en cuanto cambie la versión de PyTorch), se lee la firma real del
constructor con `inspect.signature` y se deduce qué control necesita cada campo.

Así el formulario del constructor sale gratis para los 15 optimizadores de
`torch.optim`, para el LM propio y para cualquier plugin del usuario.
"""

from __future__ import annotations

import inspect
import typing
from typing import Any

import torch

from web_interface.dominio.modelos import (
    DescriptorHiperparametro,
    DescriptorOptimizador,
)

#Optimizadores que no encajan con el bucle de entrenamiento del núcleo.
#Se listan igualmente, pero marcados, para que el usuario sepa por qué fallan.
INCOMPATIBLES: dict[str, str] = {
    "LBFGS": (
        "Necesita un `closure` en step(); train_nfs llama a optimizer.step() sin "
        "argumentos (V2_Anfis.py:442)"
    ),
    "SparseAdam": "Solo admite gradientes dispersos; los del modelo ANFIS son densos",
    "Muon": "Diseñado para matrices de peso 2D de redes profundas, no para ANFIS",
}


class IntrospectorDeOptimizadores:
    """Convierte una clase de optimizador en un `DescriptorOptimizador`."""

    #Campos que se muestran expandidos en el formulario; el resto va a "Avanzados".
    NOMBRES_PRINCIPALES = frozenset(
        {
            "lr",
            "momentum",
            "betas",
            "eps",
            "weight_decay",
            "alpha",
            "lambda_init",
            "lambda_decr",
            "lambda_incr",
        }
    )

    #Primer parámetro del constructor: lo inyecta RLANFISBuilder.Build(), no el usuario.
    PRIMEROS_IGNORADOS = frozenset({"params", "model", "modelo", "self"})

    #Los inyecta la fachada, no el formulario.
    INYECTADOS = frozenset({"device", "dispositivo"})

    def descubrir_torch(self) -> list[tuple[str, type]]:
        """Todas las subclases de `torch.optim.Optimizer` expuestas por el módulo."""
        encontrados: list[tuple[str, type]] = []
        for nombre in dir(torch.optim):
            obj = getattr(torch.optim, nombre)
            if (
                isinstance(obj, type)
                and issubclass(obj, torch.optim.Optimizer)
                and obj is not torch.optim.Optimizer
            ):
                encontrados.append((nombre, obj))
        return sorted(encontrados, key=lambda par: par[0])

    def describir(
        self,
        clase: type,
        nombre: str,
        origen: str,
        recibe_parametros: bool | None = None,
    ) -> DescriptorOptimizador:
        """Construye el descriptor completo de una clase de optimizador.

        @param recibe_parametros: si es None se deduce igual que `RLANFISBuilder.Build()`
        """
        if recibe_parametros is None:
            recibe_parametros = self._recibe_parametros(clase)

        hiperparametros: list[DescriptorHiperparametro] = []
        try:
            firma = inspect.signature(clase.__init__)
            anotaciones = typing.get_type_hints(clase.__init__)
        except (ValueError, TypeError, NameError):
            firma = None
            anotaciones = {}

        if firma is not None:
            for indice, (nombre_par, parametro) in enumerate(firma.parameters.items()):
                if indice == 0 or nombre_par in self.PRIMEROS_IGNORADOS:
                    continue
                if nombre_par in self.INYECTADOS:
                    continue
                if parametro.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                hiperparametros.append(
                    self._describir_parametro(
                        parametro, anotaciones.get(nombre_par)
                    )
                )

        motivo = INCOMPATIBLES.get(nombre)
        return DescriptorOptimizador(
            nombre=nombre,
            clase=clase,
            origen=origen,
            recibe_parametros=recibe_parametros,
            hiperparametros=hiperparametros,
            doc=self._resumen_doc(clase),
            modulo=getattr(clase, "__module__", ""),
            compatible=motivo is None,
            motivo_incompatible=motivo,
        )

    # -- internos ------------------------------------------------------------

    def _recibe_parametros(self, clase: type) -> bool:
        """Reproduce la discriminación de `RLANFISBuilder.Build()` (V2_Anfis.py:222).

        El builder pasa ``anfis.parameters()`` a los optimizadores de PyTorch y el
        modelo completo a los que derivan del `Optimizador` del núcleo.
        """
        from web_interface.nucleo import Optimizador as OptimizadorNucleo

        return not (isinstance(clase, type) and issubclass(clase, OptimizadorNucleo))

    def _describir_parametro(
        self, parametro: inspect.Parameter, anotacion: Any
    ) -> DescriptorHiperparametro:
        por_defecto = (
            None if parametro.default is inspect.Parameter.empty else parametro.default
        )
        requerido = parametro.default is inspect.Parameter.empty
        tipo, aridad, opciones = self._tipo_ui(anotacion, por_defecto)

        #Las tuplas se serializan como lista para poder viajar en JSON.
        if isinstance(por_defecto, tuple):
            por_defecto = list(por_defecto)

        return DescriptorHiperparametro(
            nombre=parametro.name,
            tipo=tipo,
            por_defecto=por_defecto,
            requerido=requerido,
            aridad=aridad,
            opciones=opciones,
            principal=parametro.name in self.NOMBRES_PRINCIPALES,
        )

    def _tipo_ui(
        self, anotacion: Any, por_defecto: Any
    ) -> tuple[str, int, list[str] | None]:
        """Decide qué control de formulario corresponde, según firma y valor por defecto."""
        texto_anotacion = self._texto(anotacion)

        #El valor por defecto es la evidencia más fiable.
        if isinstance(por_defecto, bool):
            return "bool", 1, None
        if isinstance(por_defecto, tuple):
            return "tupla", len(por_defecto), None

        #`bool | None` es el patrón de foreach/fused/capturable: tri-estado.
        if "bool" in texto_anotacion:
            if "None" in texto_anotacion or "Optional" in texto_anotacion:
                return "bool_opcional", 1, None
            return "bool", 1, None

        if "tuple" in texto_anotacion.lower():
            return "tupla", 2, None

        if "float" in texto_anotacion or "Tensor" in texto_anotacion:
            return "float", 1, None
        if "int" in texto_anotacion:
            return "int", 1, None

        if "str" in texto_anotacion:
            return "texto", 1, None

        if "Callable" in texto_anotacion:
            #No se puede escribir una función en un formulario.
            return "texto", 1, None

        #Sin anotación útil: se decide por el tipo del valor por defecto.
        if isinstance(por_defecto, int) and not isinstance(por_defecto, bool):
            return "int", 1, None
        if isinstance(por_defecto, float):
            return "float", 1, None
        if isinstance(por_defecto, str):
            return "texto", 1, None

        return "float", 1, None

    @staticmethod
    def _texto(anotacion: Any) -> str:
        if anotacion is None:
            return ""
        if isinstance(anotacion, str):
            return anotacion
        return str(anotacion)

    @staticmethod
    def _resumen_doc(clase: type) -> str:
        """Primera frase del docstring, para el tooltip del catálogo."""
        doc = inspect.getdoc(clase) or ""
        for linea in doc.splitlines():
            limpia = linea.strip()
            if limpia:
                return limpia[:200]
        return ""

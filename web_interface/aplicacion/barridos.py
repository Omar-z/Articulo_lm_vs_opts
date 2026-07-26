"""Barridos de experimentación — patrón Método Plantilla.

El barrido de experimento (`rpipeline.main`) y el de hiperparámetros
(`mejores_parametros.encontrar_parametros`) comparten alrededor del 80 % del flujo
—preparar datos, recorrer las reglas, crear el FIS, recorrer los optimizadores,
construir, entrenar y publicar progreso— y difieren solo en tres puntos: qué itera
el bucle más interno (corridas frente a valores de learning rate), qué se acumula y
qué artefacto final se produce. Hoy esos dos scripts duplican todo el esqueleto;
aquí se escribe una vez en `BarridoBase.ejecutar` y las subclases rellenan los huecos.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from web_interface.dominio.eventos import SujetoDeProgreso
from web_interface.dominio.modelos import ResultadoCorrida
from web_interface.dominio.politicas_web import (
    PoliticaCancelacion,
    PoliticaNormalizada,
    PoliticaObservador,
)
from web_interface.aplicacion.fachada_anfis import FachadaANFIS
from web_interface.configuracion import EVENTOS_POR_CORRIDA
from web_interface.nucleo import (
    DataConfig,
    DataOptimizador,
    PoliticaDeParo,
)


@dataclass(slots=True)
class VarianteDeCorrida:
    """Una repetición concreta dentro de una combinación (regla, optimizador).

    En el barrido de experimento es simplemente el número de corrida; en el de
    hiperparámetros identifica además el valor que se está probando.
    """

    indice: int
    etiqueta: str = ""
    params: dict[str, Any] | None = None


class BarridoBase(ABC):
    """Esqueleto común de los barridos.

    `ejecutar` es el método plantilla: define el orden de las operaciones y no se
    sobreescribe. Las subclases implementan los ganchos abstractos.
    """

    def __init__(
        self,
        fachada: FachadaANFIS,
        sujeto: SujetoDeProgreso,
        cancelacion: PoliticaCancelacion,
        prototipos: list[PoliticaDeParo],
        id_trabajo: str,
    ) -> None:
        self.fachada = fachada
        self.sujeto = sujeto
        self.cancelacion = cancelacion
        self.prototipos = prototipos
        self.id_trabajo = id_trabajo
        self._pasos_totales = 1
        self._pasos_hechos = 0

    # -- método plantilla ----------------------------------------------------

    def ejecutar(self, config: DataConfig) -> dict[str, Any]:
        """Recorre reglas × optimizadores × variantes. No se sobreescribe."""
        exp = config.experimentos
        entradas, salidas = self.fachada.preparar_datos(exp)

        self._pasos_totales = max(1, self._contar_pasos(config))
        self._pasos_hechos = 0

        semillas = self._semillas(config)
        acumulador = self._crear_acumulador(config)
        nombre_dataset = self.fachada.nombre_dataset(exp)

        self.sujeto.publicar(
            tipo="inicio",
            datos={
                "dataset": nombre_dataset,
                "muestras": int(entradas.shape[0]),
                "entradas": int(entradas.shape[1]),
                "pasos_totales": self._pasos_totales,
                "optimizadores": [o.nombre for o in config.optimizadores],
                "reglas": [exp.reglas_inicial, exp.reglas_total],
            },
        )

        for regla in range(exp.reglas_inicial, exp.reglas_total + 1):
            if self.cancelacion.bandera.is_set():
                return self._consolidar(acumulador, config, cancelado=True)

            ruta_fis = self.fachada.crear_fis(
                self.id_trabajo, nombre_dataset, regla, entradas, salidas, exp
            )

            for opt in config.optimizadores:
                if self.cancelacion.bandera.is_set():
                    return self._consolidar(acumulador, config, cancelado=True)

                for variante in self._variantes(config, opt, regla):
                    if self.cancelacion.bandera.is_set():
                        return self._consolidar(acumulador, config, cancelado=True)

                    resultado = self._ejecutar_corrida(
                        config, opt, regla, variante, ruta_fis,
                        entradas, salidas, semillas,
                    )
                    self._acumular(acumulador, resultado, opt, regla, variante)
                    self._pasos_hechos += 1

                self._cerrar_optimizador(acumulador, config, opt, regla)

            self.sujeto.publicar(
                tipo="regla_fin", regla=regla, fraccion_global=self._fraccion_base()
            )

        return self._consolidar(acumulador, config, cancelado=False)

    # -- una corrida ---------------------------------------------------------

    def _ejecutar_corrida(
        self,
        config: DataConfig,
        opt: DataOptimizador,
        regla: int,
        variante: VarianteDeCorrida,
        ruta_fis: str,
        entradas: np.ndarray,
        salidas: np.ndarray,
        semillas: dict[int, int],
    ) -> ResultadoCorrida:
        """Construye, entrena y evalúa una única combinación."""
        exp = config.experimentos
        contexto = self.fachada.contexto(
            regla, opt.nombre, variante.indice, exp.epocas, variante.etiqueta
        )

        #Misma semilla para todos los optimizadores en la misma corrida, para que
        #la comparación sea justa (rpipeline.py:300).
        semilla = semillas[regla] + variante.indice
        particion = self.fachada.particionar(entradas, salidas, exp, semilla)

        #La variante puede sobreescribir hiperparámetros (barrido de learning rate).
        opt_efectivo = opt
        if variante.params:
            opt_efectivo = DataOptimizador(
                nombre=opt.nombre, params={**opt.params, **variante.params}
            )

        self.sujeto.publicar(
            tipo="corrida_inicio",
            regla=regla,
            optimizador=opt.nombre,
            corrida=variante.indice,
            epocas_totales=exp.epocas,
            fraccion_global=self._fraccion_base(),
            datos={
                "variante": variante.etiqueta,
                "params": _serializable(opt_efectivo.params),
                "particion": particion.proporciones(),
                "semilla": semilla,
            },
        )

        politicas = self._politicas(contexto)
        inicio = self.fachada.cronometro()
        cancelada = False
        evaluacion: dict[str, float] = {}
        motivo = "épocas completadas"

        try:
            modelo = self.fachada.construir_modelo(ruta_fis, regla, exp, opt_efectivo)
            losses, metricas = self.fachada.entrenar(modelo, particion, exp, politicas)
            cancelada = self.cancelacion.bandera.is_set()
            if not cancelada and losses:
                evaluacion = self.fachada.evaluar(modelo, particion, exp)
            if len(losses) < exp.epocas:
                motivo = "detenido por una política de paro"
            if cancelada:
                motivo = "cancelado por el usuario"
        except Exception as exc:  # noqa: BLE001 - una corrida rota no tumba el barrido
            self.sujeto.publicar(
                tipo="error",
                regla=regla,
                optimizador=opt.nombre,
                corrida=variante.indice,
                datos={"mensaje": f"{type(exc).__name__}: {exc}"},
            )
            losses, metricas = [], {}
            motivo = f"error: {type(exc).__name__}"

        duracion = self.fachada.cronometro() - inicio
        resultado = ResultadoCorrida(
            contexto=contexto,
            losses=list(losses),
            metricas={k: list(v) for k, v in metricas.items()},
            evaluacion=evaluacion,
            motivo_paro=motivo,
            duracion=duracion,
            cancelada=cancelada,
        )

        self.sujeto.publicar(
            tipo="corrida_fin",
            regla=regla,
            optimizador=opt.nombre,
            corrida=variante.indice,
            fraccion_global=self._fraccion_base(1),
            datos={
                "variante": variante.etiqueta,
                "epocas": resultado.epocas_ejecutadas,
                "loss_final": _finito(resultado.loss_final),
                "evaluacion": {k: _finito(v) for k, v in evaluacion.items()},
                "motivo": motivo,
                "duracion": duracion,
                "cancelada": cancelada,
            },
        )
        return resultado

    def _politicas(self, contexto) -> list[PoliticaDeParo]:
        """Compone la lista de estrategias de paro de una corrida.

        El orden importa: el observador primero (siempre devuelve False, así ve
        todas las épocas), la cancelación después (para que se le atribuya el
        motivo) y al final las del núcleo, envueltas en el adaptador de tipo.

        Solo se copian las del núcleo — patrón Prototipo, igual que
        `rpipeline.py:283` — porque tienen estado mutable y cada corrida necesita
        las suyas. El observador y la cancelación se pasan por referencia: copiar
        la bandera de cancelación la dejaría sorda.
        """
        cada_n = self.fachada.epocas_por_evento(contexto.epocas, EVENTOS_POR_CORRIDA)
        observador = PoliticaObservador(
            self.sujeto,
            contexto,
            cada_n=cada_n,
            fraccion=lambda epoca: self._fraccion_base(epoca / max(1, contexto.epocas)),
        )
        return [
            observador,
            self.cancelacion,
            *[PoliticaNormalizada(p) for p in deepcopy(self.prototipos)],
        ]

    # -- progreso ------------------------------------------------------------

    def _fraccion_base(self, avance: float = 0.0) -> float:
        """Fracción global del barrido, entre 0 y 1."""
        return min(1.0, (self._pasos_hechos + avance) / self._pasos_totales)

    def _semillas(self, config: DataConfig) -> dict[int, int]:
        """Una semilla por cantidad de reglas, compartida entre optimizadores."""
        exp = config.experimentos
        generador = np.random.default_rng(getattr(config, "semilla_maestra", None))
        return {
            regla: int(generador.integers(0, 9999))
            for regla in range(exp.reglas_inicial, exp.reglas_total + 1)
        }

    def _contar_pasos(self, config: DataConfig) -> int:
        exp = config.experimentos
        reglas = exp.reglas_total - exp.reglas_inicial + 1
        por_combinacion = sum(
            len(list(self._variantes(config, opt, exp.reglas_inicial)))
            for opt in config.optimizadores
        )
        return reglas * por_combinacion

    # -- ganchos -------------------------------------------------------------

    @abstractmethod
    def _variantes(
        self, config: DataConfig, opt: DataOptimizador, regla: int
    ) -> Iterator[VarianteDeCorrida]:
        """Qué repeticiones se hacen para una combinación (regla, optimizador)."""

    @abstractmethod
    def _crear_acumulador(self, config: DataConfig) -> dict[str, Any]:
        """Estructura vacía donde se van agregando los resultados."""

    @abstractmethod
    def _acumular(
        self,
        acumulador: dict[str, Any],
        resultado: ResultadoCorrida,
        opt: DataOptimizador,
        regla: int,
        variante: VarianteDeCorrida,
    ) -> None:
        """Incorpora el resultado de una corrida al acumulador."""

    @abstractmethod
    def _consolidar(
        self, acumulador: dict[str, Any], config: DataConfig, cancelado: bool
    ) -> dict[str, Any]:
        """Calcula los agregados finales y devuelve el artefacto del barrido."""

    def _cerrar_optimizador(
        self,
        acumulador: dict[str, Any],
        config: DataConfig,
        opt: DataOptimizador,
        regla: int,
    ) -> None:
        """Gancho opcional: se llama al terminar todas las variantes de una combinación."""
        return None


# --------------------------------------------------------------------------------
# Barrido de experimento
# --------------------------------------------------------------------------------

#Los mismos 17 campos que arma `rpipeline.main:245-263`, para que el
#`resultados.json` de la web sea intercambiable con el del CLI.
def _celda_vacia() -> dict[str, Any]:
    return {
        "losses": [],
        "prom_loss": 0,
        "prom_epocas": 0,
        "r2s": [],
        "prom_r2": 0,
        "accuracys": [],
        "prom_acc": 0,
        "presicions": [],
        "prom_prec": 0,
        "SSE": [],
        "MSE": [],
        "RMSE": [],
        "MAE": [],
        "prom_SSE": 0,
        "prom_MSE": 0,
        "prom_RMSE": 0,
        "prom_MAE": 0,
    }


class BarridoDeExperimento(BarridoBase):
    """Compara optimizadores a lo largo de un rango de números de reglas.

    Equivalente a `rpipeline.main`, incluidas las estructuras de salida.
    """

    def _variantes(
        self, config: DataConfig, opt: DataOptimizador, regla: int
    ) -> Iterator[VarianteDeCorrida]:
        for corrida in range(config.experimentos.corridas):
            yield VarianteDeCorrida(indice=corrida)

    def _crear_acumulador(self, config: DataConfig) -> dict[str, Any]:
        exp = config.experimentos
        return {
            opt.nombre: {
                regla: _celda_vacia()
                for regla in range(exp.reglas_inicial, exp.reglas_total + 1)
            }
            for opt in config.optimizadores
        }

    def _acumular(
        self,
        acumulador: dict[str, Any],
        resultado: ResultadoCorrida,
        opt: DataOptimizador,
        regla: int,
        variante: VarianteDeCorrida,
    ) -> None:
        celda = acumulador[opt.nombre][regla]
        celda["losses"] += resultado.losses
        for nombre, valores in resultado.metricas.items():
            if nombre in celda:
                celda[nombre] += valores

        evaluacion = resultado.evaluacion
        if "exactitud" in evaluacion:
            celda["accuracys"].append(evaluacion["exactitud"])
            celda["presicions"].append(evaluacion["precision"])
        if "r2" in evaluacion:
            celda["r2s"].append(evaluacion["r2"])

    def _cerrar_optimizador(
        self,
        acumulador: dict[str, Any],
        config: DataConfig,
        opt: DataOptimizador,
        regla: int,
    ) -> None:
        """Promedia las corridas de esta combinación (rpipeline.py:382-393)."""
        exp = config.experimentos
        celda = acumulador[opt.nombre][regla]
        corridas = max(1, exp.corridas)

        celda["prom_loss"] = _promedio(celda["losses"])
        celda["prom_epocas"] = len(celda["losses"]) // corridas
        for metrica in ("SSE", "MSE", "RMSE", "MAE"):
            celda[f"prom_{metrica}"] = _promedio(celda[metrica])

        if exp.tipo == "clasificacion":
            celda["prom_acc"] = _promedio(celda["accuracys"])
            celda["prom_prec"] = _promedio(celda["presicions"])
        else:
            celda["prom_r2"] = _promedio(celda["r2s"])

    def _consolidar(
        self, acumulador: dict[str, Any], config: DataConfig, cancelado: bool
    ) -> dict[str, Any]:
        #Las claves numéricas se pasan a texto para poder serializar a JSON, igual
        #que hace json.dump con el diccionario de rpipeline.
        estadisticas = {
            optimizador: {str(regla): celda for regla, celda in por_regla.items()}
            for optimizador, por_regla in acumulador.items()
        }
        return {
            "tipo": "experimento",
            "cancelado": cancelado,
            "estadisticas": estadisticas,
            "terminado_en": time.time(),
        }


# --------------------------------------------------------------------------------
# Barrido de hiperparámetros
# --------------------------------------------------------------------------------

class BarridoDeHiperparametros(BarridoBase):
    """Busca el mejor learning rate de cada optimizador para cada número de reglas.

    Equivalente a `mejores_parametros.encontrar_parametros`, pero reutilizando el
    esqueleto de `BarridoBase` en vez de duplicarlo. El artefacto final tiene la
    misma forma que `mejores_parametros_optimizadores_<n>.json`, así que se puede
    usar tal cual con el CLI.

    El parámetro que se barre depende del optimizador: `lr` en los de PyTorch y
    `lambda_init` en el Levenberg-Marquardt, que es su equivalente según documenta
    el README (línea 30).
    """

    def __init__(
        self,
        *args: Any,
        valores: int = 40,
        lr_max: float = 0.99,
        lr_min: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.valores = max(2, valores)
        self.lr_max = lr_max
        self.lr_min = lr_min

    @staticmethod
    def clave_de_paso(nombre_optimizador: str) -> str:
        return "lambda_init" if nombre_optimizador == "LM" else "lr"

    def _variantes(
        self, config: DataConfig, opt: DataOptimizador, regla: int
    ) -> Iterator[VarianteDeCorrida]:
        clave = self.clave_de_paso(opt.nombre)
        rejilla = np.logspace(
            np.log10(self.lr_max), np.log10(self.lr_min), num=self.valores
        )
        for indice, valor in enumerate(rejilla):
            yield VarianteDeCorrida(
                indice=indice,
                etiqueta=f"{clave}={valor:.3e}",
                params={clave: float(valor)},
            )

    def _crear_acumulador(self, config: DataConfig) -> dict[str, Any]:
        exp = config.experimentos
        return {
            opt.nombre: {
                regla: {
                    "mejor_valor": None,
                    "mejor_loss": float("inf"),
                    "params_base": dict(opt.params),
                    "curva": [],
                }
                for regla in range(exp.reglas_inicial, exp.reglas_total + 1)
            }
            for opt in config.optimizadores
        }

    def _acumular(
        self,
        acumulador: dict[str, Any],
        resultado: ResultadoCorrida,
        opt: DataOptimizador,
        regla: int,
        variante: VarianteDeCorrida,
    ) -> None:
        celda = acumulador[opt.nombre][regla]
        clave = self.clave_de_paso(opt.nombre)
        valor = (variante.params or {}).get(clave)
        loss = resultado.loss_final

        celda["curva"].append({"valor": valor, "loss": _finito(loss)})

        #Un NaN o un infinito no puede ganar: significa que la corrida divergió.
        if np.isfinite(loss) and loss <= celda["mejor_loss"]:
            celda["mejor_loss"] = float(loss)
            celda["mejor_valor"] = valor

    def _consolidar(
        self, acumulador: dict[str, Any], config: DataConfig, cancelado: bool
    ) -> dict[str, Any]:
        #Mismo esquema que `mejores_parametros_optimizadores_*.json`:
        #{optimizador: {regla: {params con el mejor valor}}}
        parametros: dict[str, dict[str, Any]] = {}
        detalle: dict[str, dict[str, Any]] = {}

        for optimizador, por_regla in acumulador.items():
            clave = self.clave_de_paso(optimizador)
            parametros[optimizador] = {}
            detalle[optimizador] = {}
            for regla, celda in por_regla.items():
                base = dict(celda["params_base"])
                if celda["mejor_valor"] is not None:
                    base[clave] = celda["mejor_valor"]
                parametros[optimizador][str(regla)] = base
                detalle[optimizador][str(regla)] = {
                    "clave": clave,
                    "mejor_valor": celda["mejor_valor"],
                    "mejor_loss": _finito(celda["mejor_loss"]),
                    "curva": celda["curva"],
                }

        return {
            "tipo": "hiperparametros",
            "cancelado": cancelado,
            "parametros": parametros,
            "detalle": detalle,
            "terminado_en": time.time(),
        }


def _promedio(valores: list[float]) -> float:
    """Media que ignora NaN e infinitos, para que una corrida divergente no la anule."""
    if not valores:
        return 0.0
    finitos = [v for v in valores if np.isfinite(v)]
    if not finitos:
        return float("nan")
    return float(np.average(finitos))


def _finito(valor: float | None) -> float | None:
    """Sustituye NaN/inf por None: JSON no admite esos literales."""
    if valor is None:
        return None
    return float(valor) if np.isfinite(valor) else None


def _serializable(datos: dict[str, Any]) -> dict[str, Any]:
    """Deja un diccionario de hiperparámetros listo para JSON."""
    salida: dict[str, Any] = {}
    for clave, valor in datos.items():
        if isinstance(valor, (list, tuple)):
            salida[clave] = list(valor)
        elif isinstance(valor, (int, float, str, bool)) or valor is None:
            salida[clave] = valor
        else:
            salida[clave] = str(valor)
    return salida

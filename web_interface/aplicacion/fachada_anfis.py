"""Fachada sobre el núcleo ANFIS — patrón Fachada.

El propio proyecto pide esta pieza en un comentario (`V2_Anfis.py:895`):

    # Entrenar pasarlo a un patron de facade ( se lee fassad) para evitar la separacion

Una sola corrida exige coordinar seis módulos: cargar los datos, partirlos,
aplicar one-hot, crear el FIS, sanear sus NaN, construir el modelo con el builder,
moverlo al dispositivo, convertir a tensores del tipo correcto, elegir entre
`train_nfs` y `train_nfs_batch`, y evaluar con `confusion_matrix` o `PlotTraining`
según el tipo de problema. Esa secuencia está hoy duplicada casi literalmente entre
`rpipeline.main` y `mejores_parametros.encontrar_parametros`; aquí se escribe una vez.

**Fidelidad con el CLI**: se reproduce el comportamiento de `rpipeline.main`
incluso donde es discutible, para que los números sean comparables. En concreto:

* `train_size` no se usa (el CLI tampoco): la partición sale de `test_size` y
  luego `val_size` sobre el resto.
* `OneHotEncode` devuelve float32 mientras el modelo es float64
  (`funciones_auxiliares.py:97` frente a `V2_Anfis.py:33`). Se mantiene la mezcla,
  que PyTorch resuelve por promoción de tipos, salvo que se pida lo contrario.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from web_interface.dominio.modelos import ContextoDeCorrida, ParticionDatos
from web_interface.infraestructura.cargadores import FabricaDeCargadores
from web_interface.infraestructura.editor_fis import EditorDeFIS
from web_interface.infraestructura.registros import (
    NOMBRES_METRICAS,
    RegistroDeFuncionesPerdida,
    RegistroDeOptimizadores,
)
from web_interface.infraestructura.rutas import resolver_relativa
from web_interface.nucleo import (
    ANFISND,
    DataExperimento,
    DataOptimizador,
    OneHotEncode,
    PlotTraining,
    PoliticaDeParo,
    RLANFISBuilder,
    confusion_matrix,
    train_nfs,
    train_nfs_batch,
)


class ErrorDeConfiguracion(ValueError):
    """La configuración del experimento no se puede llevar a cabo."""


class FachadaANFIS:
    """Punto único de contacto con el núcleo para ejecutar corridas."""

    def __init__(
        self,
        registro_optimizadores: RegistroDeOptimizadores,
        registro_perdidas: RegistroDeFuncionesPerdida,
        fabrica_cargadores: FabricaDeCargadores,
        editor_fis: EditorDeFIS,
        dispositivo: str = "cpu",
        minilotes: bool = False,
        forzar_float64: bool = False,
    ) -> None:
        self.registro_optimizadores = registro_optimizadores
        self.registro_perdidas = registro_perdidas
        self.fabrica_cargadores = fabrica_cargadores
        self.editor_fis = editor_fis
        self.dispositivo = torch.device(dispositivo)
        self.minilotes = minilotes
        self.forzar_float64 = forzar_float64

    # -- datos ---------------------------------------------------------------

    def preparar_datos(self, exp: DataExperimento) -> tuple[np.ndarray, np.ndarray]:
        """Carga el dataset y separa características de objetivos.

        Réplica de `rpipeline.main:135-160`, con las rutas resueltas respecto a la
        raíz del repositorio en vez de al directorio de trabajo.
        """
        ruta = exp.dataset_path
        if isinstance(ruta, (list, tuple)):
            destino: Any = [resolver_relativa(r) for r in ruta]
        else:
            destino = resolver_relativa(ruta)

        opciones: dict[str, Any] = {"header": exp.dataset_header}
        if exp.dataset_sep is not None:
            opciones["sep"] = exp.dataset_sep

        datos = self.fabrica_cargadores.cargar(destino, opciones)
        if datos is None or datos.empty:
            raise ErrorDeConfiguracion(f"El dataset '{ruta}' está vacío o no se pudo leer")

        return self._separar_entradas_salidas(datos, exp)

    @staticmethod
    def _separar_entradas_salidas(
        datos: pd.DataFrame, exp: DataExperimento
    ) -> tuple[np.ndarray, np.ndarray]:
        columnas = datos.columns

        if exp.dataset_target_col is not None:
            entradas = datos.drop(columns=columnas[exp.dataset_target_col]).to_numpy()
        else:
            entradas = datos[columnas[: exp.dataset_entradas]].to_numpy()

        if exp.dataset_map_col is not None:
            if exp.dataset_target_col is None:
                raise ErrorDeConfiguracion(
                    "'dataset_map_col' requiere que 'dataset_target_col' esté definido"
                )
            objetivo = datos[columnas[exp.dataset_target_col]]
            salidas = objetivo.map(exp.dataset_map_col).to_numpy()
            if pd.isna(salidas).any():
                sin_mapear = sorted(set(objetivo[pd.isna(salidas)].astype(str)))[:5]
                raise ErrorDeConfiguracion(
                    "El mapa de clases no cubre estos valores de la columna objetivo: "
                    + ", ".join(sin_mapear)
                )
        elif exp.dataset_target_col is not None:
            salidas = datos[columnas[exp.dataset_target_col]].to_numpy()
        else:
            inicio = exp.dataset_entradas
            salidas = datos[columnas[inicio : inicio + exp.dataset_salidas]].to_numpy()

        entradas = np.asarray(entradas, dtype=np.float64)
        salidas = np.asarray(salidas)
        if salidas.ndim == 1:
            salidas = salidas.reshape(-1, 1)

        if entradas.shape[1] != exp.dataset_entradas:
            raise ErrorDeConfiguracion(
                f"Se configuraron {exp.dataset_entradas} entradas pero el dataset "
                f"aporta {entradas.shape[1]} columnas de características"
            )
        return entradas, salidas

    def particionar(
        self,
        entradas: np.ndarray,
        salidas: np.ndarray,
        exp: DataExperimento,
        semilla: int,
    ) -> ParticionDatos:
        """Divide en entrenamiento, prueba y validación, y pasa a tensores.

        Réplica exacta de `rpipeline.main:300-333`: primero se aparta `test_size`
        y después se subdivide ese resto con `val_size`. Con 0.2/0.2 el reparto
        efectivo es 80 % / 16 % / 4 %, no 60/20/20 — `train_size` no interviene.
        """
        train_x, temp_x, train_y, temp_y = train_test_split(
            entradas, salidas, test_size=exp.test_size, random_state=semilla
        )
        test_x, val_x, test_y, val_y = train_test_split(
            temp_x, temp_y, test_size=exp.val_size, random_state=semilla
        )

        #Con minilotes los datos se quedan en el host: `train_nfs_batch` sube cada
        #lote al dispositivo por separado (V2_Anfis.py:518-519).
        destino = None if self.minilotes else self.dispositivo

        tensores = [
            self._a_tensor(train_x, destino),
            self._a_tensor(test_x, destino),
            self._a_tensor(val_x, destino),
        ]
        objetivos = [
            self._objetivo(train_y, exp, destino),
            self._objetivo(test_y, exp, destino),
            self._objetivo(val_y, exp, destino),
        ]

        return ParticionDatos(
            train_x=tensores[0],
            train_y=objetivos[0],
            test_x=tensores[1],
            test_y=objetivos[1],
            val_x=tensores[2],
            val_y=objetivos[2],
            semilla=semilla,
        )

    def _a_tensor(self, datos: np.ndarray, destino: torch.device | None) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(datos)).to(torch.float64)
        return tensor if destino is None else tensor.to(destino)

    def _objetivo(
        self, datos: np.ndarray, exp: DataExperimento, destino: torch.device | None
    ) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(datos)).to(torch.float64)
        if exp.tipo == "clasificacion":
            #OneHotEncode devuelve float32; el CLI convive con esa mezcla de tipos.
            tensor = OneHotEncode(tensor, exp.dataset_salidas)
            if self.forzar_float64:
                tensor = tensor.to(torch.float64)
        return tensor if destino is None else tensor.to(destino)

    # -- modelo --------------------------------------------------------------

    def crear_fis(
        self,
        id_trabajo: str,
        nombre_dataset: str,
        regla: int,
        entradas: np.ndarray,
        salidas: np.ndarray,
        exp: DataExperimento,
    ) -> str:
        """Genera el FIS inicial de esta cantidad de reglas. Devuelve su ruta absoluta."""
        objetivo = torch.from_numpy(np.ascontiguousarray(salidas)).to(torch.float64)
        if exp.tipo == "clasificacion":
            objetivo = OneHotEncode(objetivo, exp.dataset_salidas)
        return self.editor_fis.crear_inicial(
            id_trabajo, nombre_dataset, regla, entradas, objetivo.numpy()
        )

    def construir_modelo(
        self, ruta_fis: str, regla: int, exp: DataExperimento, opt: DataOptimizador
    ) -> ANFISND:
        """Ensambla el modelo con su optimizador usando el builder del núcleo."""
        descriptor = self.registro_optimizadores.obtener(opt.nombre)
        params = dict(opt.params)

        #El LM necesita saber en qué dispositivo resolver el sistema lineal.
        #El README lo documenta como inyectado por el sistema (línea 30).
        if not descriptor.recibe_parametros:
            params["device"] = self.dispositivo

        perdida = self.registro_perdidas.obtener(exp.funcion_perdida)

        modelo = (
            RLANFISBuilder()
            .AddFIS(ruta_fis)
            .AddInputs(exp.dataset_entradas)
            .AddOutputs(exp.dataset_salidas)
            .AddRules(regla)
            .AddValMaxFails(exp.fallos)
            .AddTipoProblema(exp.tipo)
            .AddOptimizador(descriptor.clase, **params)
            .AddFunctionLoss(perdida)
            .Build()
        )

        #Salvaguarda: `train_nfs` decide cómo llamar a `step()` mirando el atributo
        #`nombre` de la instancia (V2_Anfis.py:437). Un optimizador de PyTorch que
        #se llamara "LM" recibiría `step(X, y)` y fallaría con TypeError.
        etiqueta = getattr(modelo.optimizador, "nombre", None)
        if descriptor.recibe_parametros and etiqueta == "LM":
            raise ErrorDeConfiguracion(
                f"El optimizador '{opt.nombre}' define nombre='LM', reservado para el "
                "Levenberg-Marquardt del proyecto. Renómbralo en su clase."
            )

        return modelo.to(self.dispositivo)

    # -- entrenamiento -------------------------------------------------------

    def entrenar(
        self,
        modelo: ANFISND,
        particion: ParticionDatos,
        exp: DataExperimento,
        politicas: list[PoliticaDeParo],
    ) -> tuple[list[float], dict[str, list[float]]]:
        """Ejecuta una corrida completa de entrenamiento.

        @param politicas: lista de estrategias de paro. **Nunca puede ir vacía ni
            ser un diccionario**: el valor por defecto de `train_nfs`
            (`V2_Anfis.py:401`) es el dict antiguo, y `CompositorDePoliticas`
            recorrería sus claves como si fueran políticas, fallando con
            `AttributeError: 'str' object has no attribute 'apply'`.
        """
        if not isinstance(politicas, list) or not politicas:
            raise ErrorDeConfiguracion(
                "entrenar() requiere una lista no vacía de políticas de paro"
            )

        metricas = {
            nombre: self.registro_perdidas.obtener(nombre) for nombre in NOMBRES_METRICAS
        }

        if self.minilotes:
            lote = exp.lote_size or 32
            return train_nfs_batch(
                modelo,
                particion.train_x,
                particion.train_y,
                exp.epocas,
                batch_size=lote,
                tolerancia=exp.tolerancia,
                shuffle=True,
                debug=False,
                fn_loss_lst=metricas,
                early_stop=politicas,
            )

        return train_nfs(
            modelo,
            particion.train_x,
            particion.train_y,
            exp.epocas,
            exp.tolerancia,
            debug=False,
            fn_loss_lst=metricas,
            early_stop=politicas,
        )

    # -- evaluación ----------------------------------------------------------

    def evaluar(
        self, modelo: ANFISND, particion: ParticionDatos, exp: DataExperimento
    ) -> dict[str, float]:
        """Mide el modelo sobre el conjunto de prueba.

        Clasificación devuelve exactitud y precisión macro; regresión devuelve R y R².
        """
        entradas = particion.test_x.to(self.dispositivo)
        objetivos = particion.test_y.to(self.dispositivo)

        with torch.no_grad():
            predicciones = modelo(entradas)

        try:
            if exp.tipo == "clasificacion":
                exactitud, precision = confusion_matrix(
                    predicciones,
                    objetivos,
                    num_classes=exp.dataset_salidas,
                    plot=False,
                    debug=False,
                )
                return {"exactitud": float(exactitud), "precision": float(precision)}

            #`PlotTraining` calcula un ajuste lineal auxiliar con una potencia
            #elemento a elemento ((x.T@x)**-1), que produce inf si algún término es
            #cero. Ese valor no interviene en R ni R², así que se silencian los
            #avisos de numpy en lugar de alterar el cálculo.
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                r, r2 = PlotTraining(
                    entradas, predicciones, objetivos, plot=False, debug=False
                )
            return {"r": float(r), "r2": float(r2)}
        finally:
            del predicciones, entradas, objetivos
            if self.dispositivo.type == "cuda":
                torch.cuda.empty_cache()

    # -- utilidades ----------------------------------------------------------

    def epocas_por_evento(self, epocas: int, objetivo: int) -> int:
        """Cada cuántas épocas conviene publicar un evento de progreso."""
        return max(1, epocas // max(1, objetivo))

    @staticmethod
    def nombre_dataset(exp: DataExperimento) -> str:
        """Nombre legible del dataset, como lo compone `rpipeline.main:231-232`."""
        ruta = exp.dataset_path
        if isinstance(ruta, (list, tuple)):
            ruta = ruta[0]
        return str(ruta).replace("\\", "/").split("/")[-1]

    @staticmethod
    def cronometro() -> float:
        return time.perf_counter()

    def contexto(
        self, regla: int, optimizador: str, corrida: int, epocas: int, variante: str = ""
    ) -> ContextoDeCorrida:
        return ContextoDeCorrida(
            regla=regla,
            optimizador=optimizador,
            corrida=corrida,
            epocas=epocas,
            variante=variante,
        )

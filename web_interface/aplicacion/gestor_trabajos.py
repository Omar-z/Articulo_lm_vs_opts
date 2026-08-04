"""Cola de trabajos y su hilo trabajador.

**Un solo hilo, no multiprocessing.** Las operaciones caras del núcleo
(`matmul`, `linalg.solve`, `jacfwd` en la jacobiana del LM, el `backward` de los
optimizadores de PyTorch) liberan el GIL, así que el event loop de FastAPI sigue
atendiendo peticiones mientras se entrena. Con hilos, además, la bandera de
cancelación y el emisor de eventos son memoria compartida directa; con procesos
cada evento de época exigiría serialización y en macOS el arranque `spawn`
reimportaría torch y fuzzylab en cada trabajo.

Se ejecuta **un entrenamiento a la vez** a propósito: compiten por la misma CPU y
los tiempos deben ser comparables entre optimizadores.

La contrapartida es que no hay forma de matar un trabajo de golpe. La cancelación
es cooperativa con granularidad de época, mediante `PoliticaCancelacion`.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable

from web_interface.aplicacion.comandos import Comando, ContextoDeEjecucion
from web_interface.configuracion import MAX_TRABAJOS_EN_MEMORIA
from web_interface.dominio.eventos import EventoProgreso, ObservadorDeProgreso
from web_interface.dominio.modelos import nuevo_id
from web_interface.dominio.politicas_web import PoliticaCancelacion
from web_interface.dominio.trabajo import Trabajo
from web_interface.infraestructura.salida import BitacoraDeTrabajo, MultiplexorDeSalida


class TrabajoNoEncontrado(KeyError):
    """No existe ningún trabajo con ese identificador."""


class _EspejoDeEstado:
    """Observador que mantiene al día el resumen del trabajo.

    Evita que la API tenga que recorrer el historial de eventos solo para saber en
    qué punto va el barrido.
    """

    def __init__(self, trabajo: Trabajo) -> None:
        self.trabajo = trabajo

    def notificar(self, evento: EventoProgreso) -> None:
        if evento.fraccion_global:
            self.trabajo.fraccion = evento.fraccion_global

        if evento.tipo in ("corrida_inicio", "epoca"):
            self.trabajo.paso_actual = {
                "regla": evento.regla,
                "optimizador": evento.optimizador,
                "corrida": evento.corrida,
                "epoca": evento.epoca,
                "epocas_totales": evento.epocas_totales,
                "loss": evento.loss,
            }
        elif evento.tipo == "inicio" and evento.datos:
            self.trabajo.metadatos.update(evento.datos)


class GestorDeTrabajos:
    """Registro de trabajos, cola FIFO y único hilo de ejecución."""

    def __init__(
        self,
        multiplexor: MultiplexorDeSalida | None = None,
        observadores_globales: list[ObservadorDeProgreso] | None = None,
        al_terminar: Callable[[Trabajo, dict[str, Any]], str | None] | None = None,
        maximo_en_memoria: int = MAX_TRABAJOS_EN_MEMORIA,
    ) -> None:
        self._cola: queue.Queue[tuple[Trabajo, Comando]] = queue.Queue()
        self._trabajos: dict[str, Trabajo] = {}
        self._bitacoras: dict[str, BitacoraDeTrabajo] = {}
        self._orden: list[str] = []
        self._cerrojo = threading.RLock()

        self._multiplexor = multiplexor
        self._observadores_globales = observadores_globales or []
        self._al_terminar = al_terminar
        self._maximo = maximo_en_memoria

        self._hilo: threading.Thread | None = None
        self._parar = threading.Event()
        self._trabajo_activo: str | None = None

    # -- ciclo de vida del gestor --------------------------------------------

    def iniciar(self) -> None:
        if self._hilo is not None and self._hilo.is_alive():
            return
        self._parar.clear()
        self._hilo = threading.Thread(
            target=self._bucle, name="anfis-trabajador", daemon=True
        )
        self._hilo.start()

    def detener(self, espera: float = 5.0) -> None:
        """Pide el apagado y cancela lo que esté en marcha."""
        self._parar.set()
        with self._cerrojo:
            for trabajo in self._trabajos.values():
                if not trabajo.estado.es_terminal():
                    trabajo.bandera_cancelacion.set()
        #Centinela para desbloquear el `get` del hilo.
        self._cola.put(None)  # type: ignore[arg-type]
        if self._hilo is not None:
            self._hilo.join(timeout=espera)

    # -- operaciones sobre trabajos ------------------------------------------

    def encolar(self, comando: Comando, metadatos: dict[str, Any] | None = None) -> Trabajo:
        trabajo = Trabajo(comando.id, comando.tipo, comando.descripcion())
        if metadatos:
            trabajo.metadatos.update(metadatos)

        trabajo.sujeto.suscribir(_EspejoDeEstado(trabajo))
        for observador in self._observadores_globales:
            trabajo.sujeto.suscribir(observador)

        bitacora = BitacoraDeTrabajo(
            al_recibir_linea=lambda linea, t=trabajo: t.sujeto.publicar(
                tipo="bitacora", datos={"linea": linea}
            )
        )

        with self._cerrojo:
            self._trabajos[trabajo.id] = trabajo
            self._bitacoras[trabajo.id] = bitacora
            self._orden.append(trabajo.id)
            self._podar()

        self._cola.put((trabajo, comando))
        return trabajo

    def obtener(self, id_trabajo: str) -> Trabajo:
        with self._cerrojo:
            trabajo = self._trabajos.get(id_trabajo)
        if trabajo is None:
            raise TrabajoNoEncontrado(f"No existe el trabajo '{id_trabajo}'")
        return trabajo

    def listar(self, estado: str | None = None) -> list[Trabajo]:
        with self._cerrojo:
            trabajos = [self._trabajos[i] for i in self._orden if i in self._trabajos]
        if estado:
            trabajos = [t for t in trabajos if t.estado.nombre == estado]
        return list(reversed(trabajos))

    def cancelar(self, id_trabajo: str) -> Trabajo:
        trabajo = self.obtener(id_trabajo)
        trabajo.solicitar_cancelacion()
        return trabajo

    def eliminar(self, id_trabajo: str) -> None:
        trabajo = self.obtener(id_trabajo)
        if not trabajo.estado.puede_eliminar():
            raise ValueError(
                f"Un trabajo en estado '{trabajo.estado.nombre}' no se puede eliminar"
            )
        #Si aún estaba en la cola, se marca para que el hilo lo salte al sacarlo.
        trabajo.bandera_cancelacion.set()
        with self._cerrojo:
            self._trabajos.pop(id_trabajo, None)
            self._bitacoras.pop(id_trabajo, None)
            if id_trabajo in self._orden:
                self._orden.remove(id_trabajo)

    def bitacora(self, id_trabajo: str, ultimas: int = 200) -> list[str]:
        with self._cerrojo:
            bitacora = self._bitacoras.get(id_trabajo)
        return bitacora.lineas(ultimas) if bitacora else []

    @property
    def trabajo_activo(self) -> Trabajo | None:
        with self._cerrojo:
            if self._trabajo_activo is None:
                return None
            return self._trabajos.get(self._trabajo_activo)

    def posicion_en_cola(self, id_trabajo: str) -> int:
        """Cuántos trabajos pendientes hay por delante (0 = el siguiente)."""
        pendientes = [
            t.id for t in reversed(self.listar()) if t.estado.nombre == "pendiente"
        ]
        return pendientes.index(id_trabajo) if id_trabajo in pendientes else 0

    # -- hilo trabajador -----------------------------------------------------

    def _bucle(self) -> None:
        while not self._parar.is_set():
            try:
                elemento = self._cola.get(timeout=0.5)
            except queue.Empty:
                continue

            if elemento is None:  # centinela de apagado
                break

            trabajo, comando = elemento
            try:
                self._procesar(trabajo, comando)
            finally:
                self._cola.task_done()

    def _procesar(self, trabajo: Trabajo, comando: Comando) -> None:
        #Se pudo cancelar mientras esperaba en la cola: no se arranca.
        if trabajo.bandera_cancelacion.is_set():
            if trabajo.estado.nombre == "pendiente":
                try:
                    trabajo.transicionar("cancelar")
                except Exception:  # noqa: BLE001
                    pass
            return

        bitacora = self._bitacoras.get(trabajo.id)
        if self._multiplexor is not None:
            #Desde aquí, todo lo que el núcleo imprima desde hilos secundarios
            #(incluida la barra ANSI) va a la bitácora en vez de a la consola.
            self._multiplexor.fijar_sumidero(bitacora)

        with self._cerrojo:
            self._trabajo_activo = trabajo.id

        try:
            trabajo.transicionar("arrancar")
            contexto = ContextoDeEjecucion(
                id_trabajo=trabajo.id,
                sujeto=trabajo.sujeto,
                cancelacion=PoliticaCancelacion(trabajo.bandera_cancelacion),
            )
            resultado = comando.ejecutar(contexto)
            trabajo.resultado = resultado

            if self._al_terminar is not None:
                try:
                    trabajo.ruta_resultado = self._al_terminar(trabajo, resultado)
                except Exception as exc:  # noqa: BLE001
                    trabajo.error = f"No se pudieron guardar los resultados: {exc}"

            #Un barrido cancelado conserva la fracción que alcanzó: decir 100 %
            #daría a entender que se completó.
            if not trabajo.bandera_cancelacion.is_set():
                trabajo.fraccion = 1.0
            trabajo.transicionar("terminar")
            trabajo.sujeto.publicar(
                tipo="fin",
                fraccion_global=trabajo.fraccion,
                datos={
                    "estado": trabajo.estado.nombre,
                    "ruta_resultado": trabajo.ruta_resultado,
                    "duracion": trabajo.duracion,
                    "parcial": trabajo.bandera_cancelacion.is_set(),
                },
            )

        except Exception as exc:  # noqa: BLE001 - un fallo no debe tumbar el hilo
            trabajo.error = f"{type(exc).__name__}: {exc}"
            try:
                trabajo.transicionar("fallar")
            except Exception:  # noqa: BLE001
                pass
            trabajo.sujeto.publicar(
                tipo="error", datos={"mensaje": trabajo.error, "fatal": True}
            )

        finally:
            with self._cerrojo:
                self._trabajo_activo = None
            #El sumidero NO se cierra: por los `return` prematuros de train_nfs
            #(V2_Anfis.py:466) la barra puede escribir hasta 100 ms más tarde.
            #Solo se desconecta cuando arranca el trabajo siguiente.
            if self._multiplexor is not None:
                self._multiplexor.fijar_sumidero(None)

    def _podar(self) -> None:
        """Descarta los trabajos terminados más antiguos si se supera el máximo."""
        while len(self._orden) > self._maximo:
            for indice, id_trabajo in enumerate(self._orden):
                trabajo = self._trabajos.get(id_trabajo)
                if trabajo is not None and trabajo.estado.es_terminal():
                    del self._orden[indice]
                    self._trabajos.pop(id_trabajo, None)
                    self._bitacoras.pop(id_trabajo, None)
                    break
            else:
                return  # no hay ninguno terminal que podar


def crear_id_trabajo() -> str:
    return nuevo_id()

"""Publicación de progreso — patrón Observer.

El hilo de entrenamiento produce eventos y hay tres consumidores con ritmos muy
distintos: el historial (los guarda todos, para poder reconstruir la curva tras un
F5), los WebSockets abiertos (reciben una versión diezmada) y el escritor de
resultados. Acoplar el barrido directamente a `WebSocket` lo volvería imposible de
probar sin levantar el servidor.

`SujetoDeProgreso` es seguro entre hilos: `publicar` se llama desde el hilo
trabajador y `suscribir`/`desuscribir` desde el event loop.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Protocol

from web_interface.dominio.modelos import EventoProgreso


class ObservadorDeProgreso(Protocol):
    """Cualquier interesado en los eventos de un trabajo."""

    def notificar(self, evento: EventoProgreso) -> None: ...


class SujetoDeProgreso:
    """Emisor de eventos de un trabajo concreto.

    Asigna el número de secuencia bajo un cerrojo, de modo que el orden sea total
    aunque publiquen varios hilos.
    """

    def __init__(self, id_trabajo: str) -> None:
        self.id_trabajo = id_trabajo
        self._observadores: list[ObservadorDeProgreso] = []
        self._cerrojo = threading.Lock()
        self._secuencia = 0

    def suscribir(self, observador: ObservadorDeProgreso) -> None:
        with self._cerrojo:
            if observador not in self._observadores:
                self._observadores.append(observador)

    def desuscribir(self, observador: ObservadorDeProgreso) -> None:
        with self._cerrojo:
            if observador in self._observadores:
                self._observadores.remove(observador)

    def publicar(self, tipo: str, **campos: Any) -> EventoProgreso:
        """Crea y reparte un evento. Nunca lanza: un observador roto no corta el entrenamiento."""
        with self._cerrojo:
            self._secuencia += 1
            evento = EventoProgreso(
                secuencia=self._secuencia,
                id_trabajo=self.id_trabajo,
                tipo=tipo,
                instante=time.time(),
                **campos,
            )
            observadores = list(self._observadores)

        for observador in observadores:
            try:
                observador.notificar(evento)
            except Exception:  # noqa: BLE001 - el progreso jamás debe tumbar el barrido
                pass
        return evento

    @property
    def ultima_secuencia(self) -> int:
        with self._cerrojo:
            return self._secuencia


class HistorialDeEventos:
    """Buffer circular por trabajo; fuente de verdad para reengancharse tras un F5.

    Guarda los eventos a resolución completa (acotados por `maximo`) y además
    mantiene las curvas por serie, que es lo único que necesita el monitor para
    repintarse desde cero.
    """

    def __init__(self, maximo: int = 20_000) -> None:
        self._eventos: deque[EventoProgreso] = deque(maxlen=maximo)
        self._curvas: dict[str, dict[str, list[float]]] = {}
        self._corridas_terminadas: list[dict[str, Any]] = []
        self._bitacora: list[str] = []
        self._cerrojo = threading.Lock()
        self.ultima_secuencia = 0

    def notificar(self, evento: EventoProgreso) -> None:
        with self._cerrojo:
            self._eventos.append(evento)
            self.ultima_secuencia = evento.secuencia

            if evento.tipo == "epoca" and evento.loss is not None:
                clave = evento.clave_de_curva()
                if clave is not None:
                    curva = self._curvas.setdefault(clave, {"epocas": [], "loss": []})
                    curva["epocas"].append(float(evento.epoca or 0))
                    curva["loss"].append(evento.loss)

            elif evento.tipo == "corrida_fin" and evento.datos:
                #Se guarda junto con la identidad de la corrida: el snapshot que
                #recibe el navegador tras un F5 necesita saber a qué combinación
                #pertenece cada fila, y `datos` por sí solo no lo dice.
                self._corridas_terminadas.append(
                    {
                        "optimizador": evento.optimizador,
                        "regla": evento.regla,
                        "corrida": evento.corrida,
                        **evento.datos,
                    }
                )

            elif evento.tipo == "bitacora" and evento.datos:
                linea = evento.datos.get("linea")
                if linea:
                    self._bitacora.append(str(linea))

    def desde(self, secuencia: int, limite: int = 2000) -> list[EventoProgreso]:
        """Eventos posteriores a `secuencia`, para el modo de recuperación por polling."""
        with self._cerrojo:
            return [e for e in self._eventos if e.secuencia > secuencia][:limite]

    def curvas(self, max_puntos: int = 500) -> dict[str, dict[str, list[float]]]:
        """Todas las series submuestreadas.

        Con un barrido completo esto son cientos de series: úsese solo cuando de
        verdad se quieran todas de golpe. El monitor las pide por lotes con
        `curvas_de` para poder mostrar progreso mientras carga.
        """
        with self._cerrojo:
            claves = list(self._curvas)
        return self.curvas_de(claves, max_puntos)

    def curvas_de(
        self, claves: list[str], max_puntos: int = 500
    ) -> dict[str, dict[str, list[float]]]:
        """Solo las series pedidas, submuestreadas."""
        with self._cerrojo:
            seleccion = {c: self._curvas[c] for c in claves if c in self._curvas}
        return {
            clave: {
                "epocas": submuestrear(datos["epocas"], max_puntos),
                "loss": submuestrear(datos["loss"], max_puntos),
            }
            for clave, datos in seleccion.items()
        }

    def claves_de_curvas(self) -> list[dict[str, Any]]:
        """Inventario ligero de las series: qué hay y cuánto ocupa cada una.

        Es lo que permite al navegador dibujar una barra de progreso real en vez
        de un indicador indeterminado: sabe cuántas series va a recibir antes de
        empezar a pedirlas.
        """
        with self._cerrojo:
            return [
                {"clave": clave, "puntos": len(datos["loss"])}
                for clave, datos in self._curvas.items()
            ]

    def curva(self, clave: str, max_puntos: int = 500) -> dict[str, list[float]]:
        with self._cerrojo:
            datos = self._curvas.get(clave)
            if datos is None:
                return {"epocas": [], "loss": []}
            return {
                "epocas": submuestrear(datos["epocas"], max_puntos),
                "loss": submuestrear(datos["loss"], max_puntos),
            }

    def corridas_terminadas(self) -> list[dict[str, Any]]:
        with self._cerrojo:
            return list(self._corridas_terminadas)

    def bitacora(self, ultimas: int = 200) -> list[str]:
        with self._cerrojo:
            return self._bitacora[-ultimas:]


def submuestrear(valores: list[float], max_puntos: int) -> list[float]:
    """Reduce una serie a `max_puntos` por decimación uniforme.

    Conserva siempre el primer y el último valor: el final de la curva es justo lo
    que interesa mirar, y perderlo daría una impresión falsa de dónde convergió.
    """
    total = len(valores)
    if total <= max_puntos or max_puntos < 2:
        return list(valores)

    paso = total / (max_puntos - 1)
    indices = sorted({int(i * paso) for i in range(max_puntos - 1)} | {total - 1})
    return [valores[i] for i in indices]

"""Captura de la salida por consola del núcleo.

`train_nfs` lanza un hilo demonio (`mostrar_barra_progreso`, `V2_Anfis.py:736`) que
escribe secuencias ANSI en `sys.stdout` cada 100 ms. Si eso llega a la terminal de
uvicorn, los registros de acceso quedan ilegibles.

`contextlib.redirect_stdout` capturaría la barra —el hilo resuelve `sys.stdout` en
cada iteración—, pero reemplaza `sys.stdout` de forma **global**, así que también se
tragaría los registros del servidor. La solución es instalar una sola vez un
multiplexor que enruta por hilo: el hilo principal (donde vive el event loop y por
tanto uvicorn) sigue escribiendo en la consola real, y cualquier otro hilo escribe
en la bitácora del trabajo activo.

Se enruta por «hilo principal vs. el resto» y no con `threading.local` porque el
hilo de la barra lo crea `train_nfs`, no esta aplicación, y no heredaría el valor.
"""

from __future__ import annotations

import io
import re
import sys
import threading
from collections import deque
from typing import Callable, TextIO

from web_interface.configuracion import MAX_LINEAS_BITACORA

#Secuencias de escape ANSI: colores, posicionado del cursor, borrado de pantalla.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

#`CompositorDePoliticas.apply` imprime esto al detener el entrenamiento
#(funciones_auxiliares.py:341, usando el __str__ del Protocol).
_MOTIVO_PARO = re.compile(r"Se detuvo por:\s*(.+)")


class BitacoraDeTrabajo(io.TextIOBase):
    """Buffer circular de líneas, con el ANSI ya eliminado.

    **Nunca se cierra.** Por los `return` prematuros de `train_nfs:466` y
    `train_nfs_batch:556` —que hacen `stop_event.set()` sin esperar al hilo— la
    barra puede seguir escribiendo hasta 100 ms después de que la corrida termine.
    Si el buffer estuviera cerrado, esa escritura tardía lanzaría una excepción en
    un hilo que no controlamos.
    """

    def __init__(
        self,
        maximo: int = MAX_LINEAS_BITACORA,
        al_recibir_linea: Callable[[str], None] | None = None,
    ) -> None:
        self._lineas: deque[str] = deque(maxlen=maximo)
        self._parcial = ""
        self._cerrojo = threading.Lock()
        self._al_recibir_linea = al_recibir_linea
        self.ultimo_motivo_paro: str | None = None

    def writable(self) -> bool:
        return True

    def write(self, texto: str) -> int:
        if not texto:
            return 0

        limpio = _ANSI.sub("", texto).replace("\r", "\n")
        completas: list[str] = []

        with self._cerrojo:
            self._parcial += limpio
            while "\n" in self._parcial:
                linea, self._parcial = self._parcial.split("\n", 1)
                linea = linea.rstrip()
                if linea:
                    self._lineas.append(linea)
                    completas.append(linea)

        for linea in completas:
            coincidencia = _MOTIVO_PARO.search(linea)
            if coincidencia:
                self.ultimo_motivo_paro = coincidencia.group(1).strip()
            if self._al_recibir_linea is not None:
                try:
                    self._al_recibir_linea(linea)
                except Exception:  # noqa: BLE001 - la bitácora nunca corta el barrido
                    pass

        return len(texto)

    def flush(self) -> None:
        return None

    def lineas(self, ultimas: int = 200) -> list[str]:
        with self._cerrojo:
            return list(self._lineas)[-ultimas:]

    def consumir_motivo_paro(self) -> str | None:
        """Devuelve el último motivo de paro detectado y lo olvida."""
        motivo, self.ultimo_motivo_paro = self.ultimo_motivo_paro, None
        return motivo


class MultiplexorDeSalida(io.TextIOBase):
    """`sys.stdout` de la aplicación: reparte entre la consola y la bitácora activa."""

    def __init__(self, real: TextIO) -> None:
        self._real = real
        self._hilo_principal = threading.get_ident()
        self._sumidero: BitacoraDeTrabajo | None = None

    def fijar_sumidero(self, sumidero: BitacoraDeTrabajo | None) -> None:
        """Dirige la salida de los hilos secundarios a esta bitácora."""
        self._sumidero = sumidero

    def writable(self) -> bool:
        return True

    def write(self, texto: str) -> int:
        sumidero = self._sumidero
        if sumidero is not None and threading.get_ident() != self._hilo_principal:
            return sumidero.write(texto)
        return self._real.write(texto)

    def flush(self) -> None:
        try:
            self._real.flush()
        except (ValueError, OSError):
            pass

    def isatty(self) -> bool:
        try:
            return self._real.isatty()
        except (ValueError, AttributeError):
            return False

    @property
    def real(self) -> TextIO:
        return self._real


def instalar_multiplexor() -> MultiplexorDeSalida:
    """Sustituye `sys.stdout` por el multiplexor. Idempotente."""
    if isinstance(sys.stdout, MultiplexorDeSalida):
        return sys.stdout
    multiplexor = MultiplexorDeSalida(sys.stdout)
    sys.stdout = multiplexor
    return multiplexor


def desinstalar_multiplexor() -> None:
    """Devuelve `sys.stdout` a su valor original."""
    if isinstance(sys.stdout, MultiplexorDeSalida):
        sys.stdout = sys.stdout.real

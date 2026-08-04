"""Resolución y saneado de rutas.

Regla dura del proyecto: **nada depende del directorio de trabajo**. Los JSON de
configuración referencian datasets con rutas relativas a la raíz del repositorio
(p. ej. ``"data_sets/iris/iris.data"``), y `CrearFISInicial` escribe el archivo con
el nombre que se le pasa tal cual, así que siempre se le entrega un prefijo absoluto.

Nunca se usa ``os.chdir``: es estado global del proceso y corromperia las peticiones
concurrentes del servidor.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from web_interface.configuracion import RAIZ_PROYECTO

_CARACTERES_INVALIDOS = re.compile(r"[^A-Za-z0-9._-]+")


class RutaNoPermitida(Exception):
    """Se intentó salir del directorio autorizado (path traversal)."""


def resolver_relativa(ruta: str | Path) -> Path:
    """Convierte una ruta del JSON de configuración en absoluta.

    Las rutas ya absolutas se devuelven resueltas; las relativas se interpretan
    respecto a la raíz del repositorio, no respecto al CWD.

    @param ruta: ruta tal como aparece en la configuración
    @return: ruta absoluta y normalizada
    """
    p = Path(ruta)
    if p.is_absolute():
        return p.resolve()
    return (RAIZ_PROYECTO / p).resolve()


def relativizar(ruta: str | Path) -> str:
    """Inversa de `resolver_relativa`, para exportar JSON compatible con rpipeline.py.

    Si la ruta cae dentro del repositorio devuelve la forma relativa con separadores
    POSIX; si no, devuelve la absoluta.

    Se apoya en `resolver_relativa` y **no** en `Path.resolve()` a secas: esta
    última interpreta las rutas relativas respecto al directorio de trabajo, así
    que arrancando el servidor desde otra carpeta una ruta como
    ``"data_sets/iris/iris.data"`` acabaría convertida en ``/tmp/data_sets/...``.
    """
    p = resolver_relativa(ruta)
    if p.is_relative_to(RAIZ_PROYECTO):
        return p.relative_to(RAIZ_PROYECTO).as_posix()
    return p.as_posix()


def ruta_segura(base: Path, *partes: str) -> Path:
    """Compone una ruta bajo `base` y verifica que no se escape de ella.

    @param base: directorio dentro del cual debe quedar el resultado
    @param partes: componentes a unir
    @raise RutaNoPermitida: si el resultado sale de `base`
    """
    base_resuelta = base.resolve()
    destino = base_resuelta.joinpath(*partes).resolve()
    if not destino.is_relative_to(base_resuelta):
        raise RutaNoPermitida(f"La ruta '{destino}' está fuera de '{base_resuelta}'")
    return destino


def sanear_nombre(nombre: str, por_defecto: str = "archivo") -> str:
    """Reduce un nombre de archivo a caracteres seguros.

    Descarta cualquier componente de directorio (`Path(...).name`), normaliza los
    acentos y sustituye lo que no sea alfanumérico, punto, guion o guion bajo.
    """
    base = Path(nombre).name
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode()
    base = _CARACTERES_INVALIDOS.sub("_", base).strip("._")
    return base or por_defecto


def clave_de_cache(ruta: Path) -> str:
    """Identificador estable de un archivo, sensible a su última modificación.

    Se usa para cachear el resumen de los resultados legados, que se recalcula solo
    si el archivo de origen cambia.
    """
    try:
        marca = ruta.stat().st_mtime_ns
    except OSError:
        marca = 0
    semilla = f"{ruta.as_posix()}|{marca}".encode()
    return hashlib.sha1(semilla).hexdigest()


def id_opaco(texto: str) -> str:
    """Identificador corto y estable a partir de un texto (p. ej. una ruta).

    Los datasets se direccionan por este id en la API, nunca por su ruta, para que
    ninguna URL admita `../`.
    """
    return hashlib.sha1(texto.encode()).hexdigest()[:16]

"""Motor de plantillas Jinja2 compartido por todas las vistas."""

from __future__ import annotations

from fastapi.templating import Jinja2Templates

from web_interface.configuracion import DIR_ESTATICOS, DIR_PLANTILLAS

plantillas = Jinja2Templates(directory=str(DIR_PLANTILLAS))


def url_estatico(ruta: str) -> str:
    """URL de un archivo estático con la marca de tiempo del archivo.

    Sin esto el navegador conserva la copia cacheada del CSS o del JS aunque el
    archivo haya cambiado, y las modificaciones parecen no surtir efecto: cuesta
    mucho tiempo de depuración descubrir que el código estaba bien y el navegador
    servía la versión vieja.
    """
    archivo = DIR_ESTATICOS / ruta
    try:
        version = int(archivo.stat().st_mtime)
    except OSError:
        version = 0
    return f"/estaticos/{ruta}?v={version}"


def formato_bytes(n: int | float | None) -> str:
    """Convierte un tamaño en bytes a texto legible ('1.4 MB')."""
    if n is None:
        return "—"
    tamano = float(n)
    for unidad in ("B", "KB", "MB", "GB"):
        if tamano < 1024 or unidad == "GB":
            return f"{tamano:.0f} {unidad}" if unidad == "B" else f"{tamano:.1f} {unidad}"
        tamano /= 1024
    return f"{tamano:.1f} GB"


def formato_duracion(segundos: float | None) -> str:
    """Duración legible en español, al estilo de `rpipeline.formato_legible`."""
    if segundos is None:
        return "—"
    total = int(segundos)
    horas, resto = divmod(total, 3600)
    minutos, segs = divmod(resto, 60)
    partes: list[str] = []
    if horas:
        partes.append(f"{horas} h")
    if minutos:
        partes.append(f"{minutos} min")
    if segs or not partes:
        partes.append(f"{segs} s")
    return " ".join(partes)


plantillas.env.filters["bytes"] = formato_bytes
plantillas.env.filters["duracion"] = formato_duracion
plantillas.env.globals["estatico"] = url_estatico

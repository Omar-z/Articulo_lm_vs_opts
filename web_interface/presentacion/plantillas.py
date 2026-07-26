"""Motor de plantillas Jinja2 compartido por todas las vistas."""

from __future__ import annotations

from fastapi.templating import Jinja2Templates

from web_interface.configuracion import DIR_PLANTILLAS

plantillas = Jinja2Templates(directory=str(DIR_PLANTILLAS))


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

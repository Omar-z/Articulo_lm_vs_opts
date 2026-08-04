"""Catálogo de datasets — patrón Repositorio.

Reúne dos orígenes en una sola lista: los que ya están en `data_sets/` (solo
lectura) y los que sube el usuario, que van a `data/datasets/`.

De `data_sets/` **solo se autocargan los archivos ya preprocesados**, los que
llevan el sufijo `_pre` o `-pre` antes de la extensión. El resto son datos crudos
que el modelo no puede consumir sin limpiar primero, y su presencia en el catálogo
solo estorbaba. Los archivos que sube el usuario no pasan por ese filtro.

Cada dataset se direcciona por un **identificador opaco** derivado de su ruta, no
por la ruta misma: así ninguna URL puede contener `../` y el traversal queda
descartado por construcción.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from web_interface.configuracion import (
    DIR_DATASETS_REPO,
    DIR_DATASETS_WEB,
    EXTENSIONES_DATASET,
    MAX_BYTES_DATASET,
    RAIZ_PROYECTO,
    SUFIJOS_PREPROCESADO,
)
from web_interface.infraestructura.rutas import id_opaco, sanear_nombre


class DatasetNoEncontrado(KeyError):
    """No existe el dataset pedido."""


class DatasetRechazado(ValueError):
    """El archivo no cumple los requisitos para darse de alta."""


@dataclass(slots=True)
class Dataset:
    """Un archivo de datos disponible para experimentar."""

    id: str
    nombre: str
    ruta: Path
    origen: str  # repo|web
    extension: str
    bytes: int
    modificado: float

    def a_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "nombre": self.nombre,
            "ruta_relativa": _relativa(self.ruta),
            "origen": self.origen,
            "extension": self.extension,
            "bytes": self.bytes,
            "modificado": self.modificado,
            "editable": self.origen == "web",
        }


class RepositorioDeDatasets:
    """Descubrimiento, alta y baja de datasets."""

    def __init__(
        self,
        directorio_repo: Path | None = None,
        directorio_web: Path | None = None,
    ) -> None:
        self.directorio_repo = directorio_repo or DIR_DATASETS_REPO
        self.directorio_web = directorio_web or DIR_DATASETS_WEB

    # -- consultas -----------------------------------------------------------

    def listar(self, origen: str | None = None) -> list[Dataset]:
        datasets: list[Dataset] = []
        if origen in (None, "repo"):
            datasets += self._escanear(self.directorio_repo, "repo")
        if origen in (None, "web"):
            datasets += self._escanear(self.directorio_web, "web")
        return sorted(datasets, key=lambda d: (d.origen != "web", d.nombre.lower()))

    def obtener(self, id_dataset: str) -> Dataset:
        for dataset in self.listar():
            if dataset.id == id_dataset:
                return dataset
        raise DatasetNoEncontrado(f"No existe el dataset '{id_dataset}'")

    def emparejar_mat(self, id_dataset: str) -> Dataset | None:
        """Busca la pareja de un `.mat` de entradas o de objetivos.

        El dataset `engine` viene partido en `engineInputs.mat` y
        `engineTargets.mat`; el constructor necesita ofrecerlos juntos.
        """
        dataset = self.obtener(id_dataset)
        if dataset.extension != ".mat":
            return None

        base = dataset.ruta.stem.lower()
        pares = (("inputs", "targets"), ("input", "target"), ("in", "out"), ("x", "y"))
        for primero, segundo in pares:
            if base.endswith(primero):
                candidato = base[: -len(primero)] + segundo
            elif base.endswith(segundo):
                candidato = base[: -len(segundo)] + primero
            else:
                continue
            for otro in self.listar():
                if otro.extension == ".mat" and otro.ruta.stem.lower() == candidato:
                    return otro
        return None

    # -- altas y bajas -------------------------------------------------------

    def dar_de_alta(self, nombre: str, flujo: BinaryIO) -> Dataset:
        """Guarda un archivo subido. Solo acepta extensiones de datos.

        Estos archivos **nunca se importan como código**: se leen con pandas o
        scipy, así que subirlos no ejecuta nada.
        """
        limpio = sanear_nombre(nombre, "dataset")
        extension = Path(limpio).suffix.lower()
        if extension not in EXTENSIONES_DATASET:
            raise DatasetRechazado(
                f"La extensión '{extension}' no está permitida. "
                f"Admitidas: {', '.join(sorted(EXTENSIONES_DATASET))}"
            )

        self.directorio_web.mkdir(parents=True, exist_ok=True)
        destino = self.directorio_web / limpio

        #Si ya existe, se numera en vez de sobreescribir.
        contador = 1
        while destino.exists():
            destino = self.directorio_web / f"{Path(limpio).stem}_{contador}{extension}"
            contador += 1

        escritos = 0
        try:
            with open(destino, "wb") as archivo:
                while trozo := flujo.read(1024 * 1024):
                    escritos += len(trozo)
                    if escritos > MAX_BYTES_DATASET:
                        raise DatasetRechazado(
                            f"El archivo supera el máximo de "
                            f"{MAX_BYTES_DATASET // (1024 * 1024)} MB"
                        )
                    archivo.write(trozo)
        except DatasetRechazado:
            destino.unlink(missing_ok=True)
            raise

        if escritos == 0:
            destino.unlink(missing_ok=True)
            raise DatasetRechazado("El archivo está vacío")

        return self._describir(destino, "web")

    def eliminar(self, id_dataset: str) -> None:
        """Borra un dataset subido. Los de `data_sets/` no se tocan."""
        dataset = self.obtener(id_dataset)
        if dataset.origen != "web":
            raise DatasetRechazado(
                "Los datasets del repositorio son de solo lectura; "
                "solo se pueden borrar los que se han subido"
            )
        dataset.ruta.unlink(missing_ok=True)

    def copiar_desde(self, ruta: Path) -> Dataset:
        """Copia un archivo existente al espacio de la web."""
        self.directorio_web.mkdir(parents=True, exist_ok=True)
        destino = self.directorio_web / sanear_nombre(ruta.name)
        shutil.copy2(ruta, destino)
        return self._describir(destino, "web")

    # -- internos ------------------------------------------------------------

    def _escanear(self, directorio: Path, origen: str) -> list[Dataset]:
        if not directorio.exists():
            return []
        encontrados: list[Dataset] = []
        for ruta in sorted(directorio.rglob("*")):
            if not ruta.is_file():
                continue
            if ruta.suffix.lower() not in EXTENSIONES_DATASET:
                continue
            if ruta.name.startswith("."):
                continue
            #Del repositorio solo se autocargan los archivos ya preprocesados; los
            #que sube el usuario se muestran siempre, porque los sube a propósito.
            if origen == "repo" and not es_preprocesado(ruta):
                continue
            try:
                encontrados.append(self._describir(ruta, origen))
            except OSError:
                continue
        return encontrados

    @staticmethod
    def _describir(ruta: Path, origen: str) -> Dataset:
        estado = ruta.stat()
        return Dataset(
            id=id_opaco(str(ruta.resolve())),
            nombre=ruta.name,
            ruta=ruta.resolve(),
            origen=origen,
            extension=ruta.suffix.lower(),
            bytes=estado.st_size,
            modificado=estado.st_mtime,
        )


def es_preprocesado(ruta: Path) -> bool:
    """Indica si el archivo está marcado como listo para entrenar.

    Se reconoce por el sufijo del nombre, antes de la extensión: `abalone_pre.csv`
    o `wine-pre.csv`. Los datos crudos del repositorio necesitan limpieza previa
    (fechas a Unix, clases renumeradas desde 0, nulos) y no se pueden entrenar tal
    cual, así que no se ofrecen en el catálogo.
    """
    return ruta.stem.lower().endswith(SUFIJOS_PREPROCESADO)


def _relativa(ruta: Path) -> str:
    """Ruta relativa a la raíz del repositorio, como la escribe el JSON de configuración."""
    try:
        return ruta.relative_to(RAIZ_PROYECTO).as_posix()
    except ValueError:
        return ruta.as_posix()

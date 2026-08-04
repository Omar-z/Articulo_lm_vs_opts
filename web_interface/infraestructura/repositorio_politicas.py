"""Políticas de paro del usuario: lectura, validación y escritura del archivo.

El usuario escribe la clase en la página `/politicas` y aquí se comprueba, se
añade a `politicas/politicas_usuario.py` y se carga en memoria.

**Aviso de seguridad.** Esto ejecuta código Python que llega por HTTP, que es lo
más delicado de toda la aplicación. Las mitigaciones son proporcionadas a una
herramienta local de investigación:

* Solo se permite escribir con el servidor escuchando en `127.0.0.1`. Si se arranca
  con `--exponer`, la escritura se rechaza (la lectura sigue funcionando).
* El código se valida con `ast` **antes** de tocar el disco: tiene que ser sintaxis
  correcta y definir exactamente la clase esperada, con un método `apply`.
* Antes de guardarla se instancia y se prueba en un módulo aislado. Una política
  que falla en la prueba no llega al archivo, y así no rompe un barrido de horas.
* Cada guardado deja una copia del archivo anterior en `data/cache`, para poder
  volver atrás si algo queda mal.

No se intenta un aislamiento real (subproceso restringido, lista blanca de AST):
sería una falsa sensación de seguridad y estorbaría a políticas legítimas que usen
numpy o torch.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
import shutil
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from web_interface.configuracion import ARCHIVO_POLITICAS, DIR_CACHE

#Nombre del módulo con el que se registra el archivo del usuario.
_MODULO = "web_interface_politicas_usuario"


class PoliticaRechazada(ValueError):
    """El código de la política no cumple el contrato."""


class EscrituraNoPermitida(PermissionError):
    """No se puede escribir el archivo de políticas en esta configuración."""


@dataclass(slots=True)
class PoliticaDisponible:
    """Una política que se puede usar en un experimento."""

    nombre_clase: str
    etiqueta: str
    origen: str  # nucleo|usuario
    doc: str = ""
    editable: bool = False
    codigo: str = ""
    parametros: list[dict[str, Any]] = field(default_factory=list)

    def a_dict(self) -> dict[str, Any]:
        return {
            "nombre_clase": self.nombre_clase,
            "etiqueta": self.etiqueta,
            "origen": self.origen,
            "doc": self.doc,
            "editable": self.editable,
            "codigo": self.codigo,
            "parametros": self.parametros,
        }


def nombre_a_clase(nombre: str) -> str:
    """Convierte un nombre legible en un identificador de clase válido.

    ``"Politica Promedio Tolerancia"`` -> ``"PoliticaPromedioTolerancia"``
    """
    limpio = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode()
    partes = re.findall(r"[A-Za-z0-9]+", limpio)
    if not partes:
        raise PoliticaRechazada(
            "El nombre debe tener al menos una letra o número"
        )
    #Se respeta el uso de mayúsculas de cada palabra si ya venía en CamelCase.
    clase = "".join(p if p[:1].isupper() else p.capitalize() for p in partes)
    if clase[0].isdigit():
        raise PoliticaRechazada("El nombre no puede empezar por un número")
    return clase


def plantilla_de_politica(nombre: str) -> str:
    """Esqueleto que la página muestra por defecto en el editor."""
    clase = nombre_a_clase(nombre) if nombre.strip() else "MiPolitica"
    etiqueta = nombre.strip() or "Mi política"
    return (
        f'class {clase}(PoliticaDeParo):\n'
        f'    """Describe aquí cuándo debe detenerse el entrenamiento."""\n'
        f'\n'
        f'    def __init__(self) -> None:\n'
        f'        self.nombre = "{etiqueta}"\n'
        f'\n'
        f'    def apply(self, loss_value: float, optimizador: Any) -> bool:\n'
        f'        #Se llama una vez por época. Devuelve True para detener la corrida.\n'
        f'        return False\n'
    )


class RepositorioDePoliticas:
    """Gestiona el archivo de políticas del usuario."""

    def __init__(
        self, archivo: Path | None = None, permitir_escritura: bool = True
    ) -> None:
        self.archivo = archivo or ARCHIVO_POLITICAS
        self.permitir_escritura = permitir_escritura
        self._modulo: ModuleType | None = None
        self.error_de_carga: str | None = None

    # -- lectura -------------------------------------------------------------

    def cargar(self) -> tuple[list[PoliticaDisponible], str | None]:
        """Importa el archivo y devuelve las políticas que define, más el error si lo hubo.

        Se recarga de cero cada vez: el usuario puede haber editado el archivo a
        mano entre una llamada y otra.
        """
        self.error_de_carga = None
        if not self.archivo.exists():
            return [], None

        try:
            modulo = self._importar()
        except Exception as exc:  # noqa: BLE001 - un archivo roto no tumba el servidor
            self.error_de_carga = f"{type(exc).__name__}: {exc}"
            return [], self.error_de_carga

        self._modulo = modulo
        codigos = self._codigos_por_clase()

        politicas: list[PoliticaDisponible] = []
        for nombre, clase in self._clases_validas(modulo):
            politicas.append(
                PoliticaDisponible(
                    nombre_clase=nombre,
                    etiqueta=self._etiqueta_de(clase, nombre),
                    origen="usuario",
                    doc=(inspect.getdoc(clase) or "").split("\n")[0],
                    editable=True,
                    codigo=codigos.get(nombre, ""),
                    parametros=self._parametros_de(clase),
                )
            )
        return politicas, None

    def clase(self, nombre_clase: str) -> type:
        """Devuelve la clase ya cargada."""
        if self._modulo is None:
            self.cargar()
        clase = getattr(self._modulo, nombre_clase, None) if self._modulo else None
        if clase is None:
            raise KeyError(f"No existe la política '{nombre_clase}'")
        return clase

    def codigo_completo(self) -> str:
        """Contenido íntegro del archivo, para mostrarlo o descargarlo."""
        try:
            return self.archivo.read_text(encoding="utf-8")
        except OSError:
            return ""

    # -- validación ----------------------------------------------------------

    def validar(self, nombre_clase: str, codigo: str) -> list[str]:
        """Comprueba el código sin escribirlo. Devuelve la lista de problemas."""
        problemas: list[str] = []

        try:
            arbol = ast.parse(codigo)
        except SyntaxError as exc:
            linea = exc.lineno or 1
            return [f"Error de sintaxis en la línea {linea}: {exc.msg}"]

        clases = [n for n in arbol.body if isinstance(n, ast.ClassDef)]
        if not clases:
            return ["El código debe definir una clase"]
        if len(clases) > 1:
            problemas.append(
                "Define solo una clase por política; se encontraron: "
                + ", ".join(c.name for c in clases)
            )

        definida = clases[0]
        if definida.name != nombre_clase:
            problemas.append(
                f"La clase se llama '{definida.name}' pero el nombre indicado "
                f"produce '{nombre_clase}'. Deben coincidir."
            )

        metodos = {n.name for n in definida.body if isinstance(n, ast.FunctionDef)}
        if "apply" not in metodos:
            problemas.append(
                "Falta el método 'apply(self, loss_value, optimizador) -> bool', "
                "que es el que se llama en cada época"
            )

        #Nombres del núcleo que no se pueden pisar sin romper el despacho interno.
        if nombre_clase in ("PoliticaDeParo", "CompositorDePoliticas"):
            problemas.append(f"'{nombre_clase}' es un nombre reservado del núcleo")

        return problemas

    def probar(self, nombre_clase: str, codigo: str) -> dict[str, Any]:
        """Instancia la política en un módulo aislado y la ejecuta unas cuantas veces.

        Detecta aquí los fallos que, de otro modo, aparecerían a mitad de un
        barrido de horas.
        """
        modulo = ModuleType(f"{_MODULO}_prueba")
        modulo.__dict__.update(self._contexto_de_ejecucion())

        try:
            exec(compile(codigo, "<politica>", "exec"), modulo.__dict__)
        except Exception as exc:  # noqa: BLE001
            raise PoliticaRechazada(
                f"El código no se pudo ejecutar: {type(exc).__name__}: {exc}"
            ) from exc

        clase = getattr(modulo, nombre_clase, None)
        if clase is None:
            raise PoliticaRechazada(f"El código no define la clase '{nombre_clase}'")

        try:
            instancia = clase()
        except Exception as exc:  # noqa: BLE001
            raise PoliticaRechazada(
                f"No se pudo crear la política sin argumentos: "
                f"{type(exc).__name__}: {exc}. Da un valor por defecto a todos "
                f"los parámetros de __init__."
            ) from exc

        if not hasattr(instancia, "nombre"):
            raise PoliticaRechazada(
                "La política debe fijar 'self.nombre' en __init__: es lo que se "
                "muestra al detener el entrenamiento"
            )

        #Secuencia de pérdidas decreciente, como en un entrenamiento real, más un
        #caso degenerado al final.
        resultados: list[bool] = []
        try:
            for valor in (100.0, 10.0, 1.0, 0.1, 0.01, 1e-9, 0.0):
                resultados.append(bool(instancia.apply(valor, None)))
        except Exception as exc:  # noqa: BLE001
            raise PoliticaRechazada(
                f"'apply' falló con una pérdida de prueba: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        return {
            "nombre": str(getattr(instancia, "nombre")),
            "resultados": resultados,
            "detiene_siempre": all(resultados),
            "nunca_detiene": not any(resultados),
        }

    # -- escritura -----------------------------------------------------------

    def guardar(self, nombre_clase: str, codigo: str) -> PoliticaDisponible:
        """Añade o reemplaza una política en el archivo.

        Valida y prueba antes de tocar el disco, y guarda una copia de seguridad
        del archivo anterior.
        """
        if not self.permitir_escritura:
            raise EscrituraNoPermitida(
                "La escritura de políticas está desactivada porque el servidor no "
                "es local. Arranca sin '--exponer' para poder editarlas."
            )

        problemas = self.validar(nombre_clase, codigo)
        if problemas:
            raise PoliticaRechazada(" · ".join(problemas))

        prueba = self.probar(nombre_clase, codigo)

        contenido = self._contenido_actual()
        bloques = self._bloques_de_clases(contenido)

        cuerpo = codigo.strip("\n")
        if nombre_clase in bloques:
            inicio, fin = bloques[nombre_clase]
            lineas = contenido.splitlines()
            nuevas = lineas[:inicio] + cuerpo.splitlines() + lineas[fin:]
            nuevo = "\n".join(nuevas).rstrip() + "\n"
        else:
            nuevo = contenido.rstrip() + "\n\n\n" + cuerpo + "\n"

        #Comprobación final sobre el archivo entero: si el añadido rompiera el
        #módulo, ninguna política se podría cargar.
        try:
            ast.parse(nuevo)
        except SyntaxError as exc:
            raise PoliticaRechazada(
                f"El archivo resultante no es válido: línea {exc.lineno}: {exc.msg}"
            ) from exc

        self._respaldar()
        self.archivo.parent.mkdir(parents=True, exist_ok=True)
        self.archivo.write_text(nuevo, encoding="utf-8")

        return PoliticaDisponible(
            nombre_clase=nombre_clase,
            etiqueta=prueba["nombre"],
            origen="usuario",
            editable=True,
            codigo=cuerpo,
        )

    def eliminar(self, nombre_clase: str) -> None:
        """Quita una política del archivo."""
        if not self.permitir_escritura:
            raise EscrituraNoPermitida(
                "La escritura de políticas está desactivada porque el servidor no "
                "es local."
            )

        contenido = self._contenido_actual()
        bloques = self._bloques_de_clases(contenido)
        if nombre_clase not in bloques:
            raise KeyError(f"No existe la política '{nombre_clase}'")

        inicio, fin = bloques[nombre_clase]
        lineas = contenido.splitlines()
        nuevo = "\n".join(lineas[:inicio] + lineas[fin:]).rstrip() + "\n"

        self._respaldar()
        self.archivo.write_text(nuevo, encoding="utf-8")

    # -- internos ------------------------------------------------------------

    def _contenido_actual(self) -> str:
        if self.archivo.exists():
            return self.archivo.read_text(encoding="utf-8")
        return _CABECERA

    def _respaldar(self) -> None:
        if not self.archivo.exists():
            return
        try:
            DIR_CACHE.mkdir(parents=True, exist_ok=True)
            copia = DIR_CACHE / f"politicas_{int(time.time())}.py.bak"
            shutil.copy2(self.archivo, copia)
        except OSError:
            pass  # el respaldo es una comodidad, no un requisito

    def _importar(self) -> ModuleType:
        """Carga el archivo como módulo, descartando cualquier versión previa."""
        sys.modules.pop(_MODULO, None)
        especificacion = importlib.util.spec_from_file_location(_MODULO, self.archivo)
        if especificacion is None or especificacion.loader is None:
            raise ImportError(f"No se pudo preparar la carga de '{self.archivo.name}'")

        modulo = importlib.util.module_from_spec(especificacion)
        sys.modules[_MODULO] = modulo
        try:
            especificacion.loader.exec_module(modulo)
        except Exception:
            sys.modules.pop(_MODULO, None)
            raise
        return modulo

    @staticmethod
    def _contexto_de_ejecucion() -> dict[str, Any]:
        """Nombres disponibles para el código del usuario, como en la cabecera del archivo."""
        import numpy
        import torch

        from web_interface.nucleo import PoliticaDeParo

        return {
            "Any": Any,
            "np": numpy,
            "numpy": numpy,
            "torch": torch,
            "PoliticaDeParo": PoliticaDeParo,
        }

    @staticmethod
    def _clases_validas(modulo: ModuleType) -> list[tuple[str, type]]:
        """Clases del módulo que sirven como política de paro."""
        from web_interface.nucleo import PoliticaDeParo

        validas: list[tuple[str, type]] = []
        for nombre, objeto in inspect.getmembers(modulo, inspect.isclass):
            if objeto is PoliticaDeParo:
                continue
            #Definida aquí, no importada: si no, `PoliticaTolerancia` y compañía
            #aparecerían como políticas del usuario por estar importadas.
            if getattr(objeto, "__module__", "") != modulo.__name__:
                continue
            if not hasattr(objeto, "apply"):
                continue
            validas.append((nombre, objeto))
        return validas

    @staticmethod
    def _etiqueta_de(clase: type, por_defecto: str) -> str:
        """Nombre legible: el que la política se pone a sí misma en `__init__`."""
        try:
            return str(getattr(clase(), "nombre", por_defecto))
        except Exception:  # noqa: BLE001 - si no se puede instanciar, se usa el de la clase
            return por_defecto

    @staticmethod
    def _parametros_de(clase: type) -> list[dict[str, Any]]:
        """Parámetros configurables del constructor, para mostrarlos en la ficha."""
        try:
            firma = inspect.signature(clase.__init__)
        except (ValueError, TypeError):
            return []
        parametros = []
        for indice, (nombre, parametro) in enumerate(firma.parameters.items()):
            if indice == 0 or nombre == "self":
                continue
            parametros.append(
                {
                    "nombre": nombre,
                    "por_defecto": (
                        None
                        if parametro.default is inspect.Parameter.empty
                        else _serializable(parametro.default)
                    ),
                }
            )
        return parametros

    def _codigos_por_clase(self) -> dict[str, str]:
        """Código fuente de cada clase, para poder editarla en la página."""
        contenido = self._contenido_actual()
        lineas = contenido.splitlines()
        return {
            nombre: "\n".join(lineas[inicio:fin]).rstrip()
            for nombre, (inicio, fin) in self._bloques_de_clases(contenido).items()
        }

    @staticmethod
    def _bloques_de_clases(contenido: str) -> dict[str, tuple[int, int]]:
        """Rango de líneas (0-based, fin exclusivo) de cada clase del archivo.

        Se localiza con `ast` y no con expresiones regulares para que un decorador,
        un docstring con la palabra `class` o una clase anidada no lo despisten.
        """
        try:
            arbol = ast.parse(contenido)
        except SyntaxError:
            return {}

        bloques: dict[str, tuple[int, int]] = {}
        for nodo in arbol.body:
            if not isinstance(nodo, ast.ClassDef):
                continue
            inicio = (nodo.decorator_list[0].lineno - 1) if nodo.decorator_list else (
                nodo.lineno - 1
            )
            fin = nodo.end_lineno or nodo.lineno
            bloques[nodo.name] = (inicio, fin)
        return bloques


def _serializable(valor: Any) -> Any:
    if isinstance(valor, (str, int, float, bool)) or valor is None:
        return valor
    if isinstance(valor, (list, tuple)):
        return list(valor)
    return str(valor)


_CABECERA = '''"""Políticas de paro definidas por el usuario.

Este archivo lo escribe la página `/politicas` de la interfaz web, pero también se
puede editar a mano: al pulsar «Recargar» se vuelve a leer entero.
"""

from typing import Any

import numpy as np
import torch  # noqa: F401

from funciones_auxiliares import PoliticaDeParo
'''

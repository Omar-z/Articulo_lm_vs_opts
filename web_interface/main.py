"""Punto de entrada de la interfaz web.

Uso:
    python -m web_interface.main
    python -m web_interface.main --puerto 8080
    python -m web_interface.main --exponer      # escucha en todas las interfaces

Se puede arrancar desde cualquier directorio: `configuracion` resuelve la raíz del
proyecto a partir de la ubicación de este archivo.
"""

from __future__ import annotations

import web_interface.configuracion as configuracion  # noqa: F401  (fija MPLBACKEND)

import argparse
import os

import matplotlib
import torch
import uvicorn

from web_interface.configuracion import HOST, PUERTO

#Backend sin ventana: el núcleo dibuja con pyplot desde el hilo trabajador.
matplotlib.use("Agg")


def _analizar_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interfaz web del laboratorio ANFIS")
    parser.add_argument("--puerto", type=int, default=PUERTO, help="Puerto de escucha")
    parser.add_argument(
        "--exponer",
        action="store_true",
        help="Escuchar en 0.0.0.0 en vez de solo en localhost",
    )
    parser.add_argument(
        "--hilos",
        type=int,
        default=None,
        help="Hilos de PyTorch (por defecto, núcleos disponibles menos uno)",
    )
    return parser.parse_args()


def main() -> None:
    args = _analizar_argumentos()

    #Se deja un núcleo libre para que el event loop siga atendiendo peticiones
    #mientras el hilo trabajador entrena.
    hilos = args.hilos or max(1, (os.cpu_count() or 2) - 1)
    torch.set_num_threads(hilos)

    host = "0.0.0.0" if args.exponer else HOST
    if args.exponer:
        #Con el servidor abierto a la red se desactiva la edición de políticas:
        #escribirlas equivale a ejecutar código en este proceso.
        configuracion.EXPUESTO_EN_RED = True
        print(
            "\033[1;31m[AVISO] El servidor queda accesible desde la red. "
            "La edición de políticas queda desactivada, y los archivos de "
            "plugins/ se ejecutan con los permisos de este proceso.\033[0m"
        )

    print(f"Laboratorio ANFIS en http://{host}:{args.puerto}  ({hilos} hilos de torch)")
    uvicorn.run(
        "web_interface.presentacion.app:app",
        host=host,
        port=args.puerto,
        log_level="info",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Genera un dataset de detección sintético para probar el circuito sin descargar nada.

Dibuja figuras de colores sobre un fondo con ruido y escribe las etiquetas en formato
YOLO. Sirve para comprobar que el entrenamiento, la métrica mAP y la exportación
funcionan antes de invertir tiempo en etiquetar datos reales.

    python preparacion/generar_deteccion_sintetica.py --train 200 --val 50
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RAIZ = Path(__file__).resolve().parent.parent
CLASES = ["circulo", "cuadrado", "triangulo"]
COLORES = [(220, 70, 70), (70, 140, 220), (90, 200, 120)]


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset de detección sintético")
    p.add_argument("--destino", type=Path, default=RAIZ / "datos_deteccion")
    p.add_argument("--train", type=int, default=200)
    p.add_argument("--val", type=int, default=50)
    p.add_argument("--tam", type=int, default=320)
    p.add_argument("--max-objetos", type=int, default=4)
    p.add_argument("--semilla", type=int, default=42)
    return p.parse_args()


def figura(dibujo: ImageDraw.ImageDraw, clase: int, caja, color) -> None:
    x1, y1, x2, y2 = caja
    if clase == 0:
        dibujo.ellipse([x1, y1, x2, y2], fill=color)
    elif clase == 1:
        dibujo.rectangle([x1, y1, x2, y2], fill=color)
    else:
        dibujo.polygon([(x1, y2), ((x1 + x2) / 2, y1), (x2, y2)], fill=color)


def generar(destino: Path, cuantas: int, tam: int, max_objetos: int, aleatorio) -> None:
    dir_img = destino / "imagenes"
    dir_lab = destino / "etiquetas"
    dir_img.mkdir(parents=True, exist_ok=True)
    dir_lab.mkdir(parents=True, exist_ok=True)

    for indice in range(cuantas):
        fondo = np.random.default_rng(aleatorio.randint(0, 10**6)).integers(
            30, 70, (tam, tam, 3), dtype=np.uint8)
        imagen = Image.fromarray(fondo)
        dibujo = ImageDraw.Draw(imagen)
        lineas = []

        for _ in range(aleatorio.randint(1, max_objetos)):
            clase = aleatorio.randrange(len(CLASES))
            lado = aleatorio.randint(tam // 8, tam // 3)
            x1 = aleatorio.randint(0, tam - lado - 1)
            y1 = aleatorio.randint(0, tam - lado - 1)
            caja = (x1, y1, x1 + lado, y1 + lado)
            figura(dibujo, clase, caja, COLORES[clase])
            cx, cy = (x1 + lado / 2) / tam, (y1 + lado / 2) / tam
            lineas.append(f"{clase} {cx:.6f} {cy:.6f} {lado / tam:.6f} {lado / tam:.6f}")

        imagen.save(dir_img / f"{indice:05d}.jpg", quality=92)
        (dir_lab / f"{indice:05d}.txt").write_text("\n".join(lineas))


def main() -> None:
    args = argumentos()
    aleatorio = random.Random(args.semilla)
    (args.destino).mkdir(parents=True, exist_ok=True)
    (args.destino / "clases.txt").write_text("\n".join(CLASES) + "\n")

    generar(args.destino / "train", args.train, args.tam, args.max_objetos, aleatorio)
    generar(args.destino / "val", args.val, args.tam, args.max_objetos, aleatorio)

    print(f"{args.train} imágenes de entrenamiento y {args.val} de validación en {args.destino}")
    print(f"Clases: {', '.join(CLASES)}")
    print("\nAhora:  python entrenar.py --config configs/deteccion.yaml "
          "--set entrenamiento.epocas=3")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Genera un dataset de segmentación sintético (imagen + máscara) para probar el circuito.

    python preparacion/generar_segmentacion_sintetica.py --train 120 --val 30
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RAIZ = Path(__file__).resolve().parent.parent
CLASES = ["fondo", "circulo", "cuadrado", "triangulo"]
COLORES = [None, (220, 70, 70), (70, 140, 220), (90, 200, 120)]


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset de segmentación sintético")
    p.add_argument("--destino", type=Path, default=RAIZ / "datos_segmentacion")
    p.add_argument("--train", type=int, default=120)
    p.add_argument("--val", type=int, default=30)
    p.add_argument("--tam", type=int, default=256)
    p.add_argument("--semilla", type=int, default=42)
    return p.parse_args()


def dibujar(dibujo, clase: int, caja, relleno) -> None:
    x1, y1, x2, y2 = caja
    if clase == 1:
        dibujo.ellipse([x1, y1, x2, y2], fill=relleno)
    elif clase == 2:
        dibujo.rectangle([x1, y1, x2, y2], fill=relleno)
    else:
        dibujo.polygon([(x1, y2), ((x1 + x2) / 2, y1), (x2, y2)], fill=relleno)


def generar(destino: Path, cuantas: int, tam: int, aleatorio) -> None:
    dir_img, dir_mask = destino / "imagenes", destino / "mascaras"
    dir_img.mkdir(parents=True, exist_ok=True)
    dir_mask.mkdir(parents=True, exist_ok=True)

    for indice in range(cuantas):
        fondo = np.random.default_rng(aleatorio.randint(0, 10**6)).integers(
            30, 70, (tam, tam, 3), dtype=np.uint8)
        imagen = Image.fromarray(fondo)
        mascara = Image.new("L", (tam, tam), 0)
        pincel_img, pincel_mask = ImageDraw.Draw(imagen), ImageDraw.Draw(mascara)

        for _ in range(aleatorio.randint(1, 3)):
            clase = aleatorio.randint(1, 3)
            lado = aleatorio.randint(tam // 6, tam // 2)
            x1 = aleatorio.randint(0, tam - lado - 1)
            y1 = aleatorio.randint(0, tam - lado - 1)
            caja = (x1, y1, x1 + lado, y1 + lado)
            dibujar(pincel_img, clase, caja, COLORES[clase])
            dibujar(pincel_mask, clase, caja, clase)      # el valor del píxel ES la clase

        imagen.save(dir_img / f"{indice:05d}.jpg", quality=92)
        mascara.save(dir_mask / f"{indice:05d}.png")


def main() -> None:
    args = argumentos()
    aleatorio = random.Random(args.semilla)
    args.destino.mkdir(parents=True, exist_ok=True)
    (args.destino / "clases.txt").write_text("\n".join(CLASES) + "\n")
    generar(args.destino / "train", args.train, args.tam, aleatorio)
    generar(args.destino / "val", args.val, args.tam, aleatorio)
    print(f"{args.train} + {args.val} pares imagen/máscara en {args.destino}")
    print("Clases:", ", ".join(CLASES))
    print("\nAhora:  python entrenar.py --config configs/segmentacion.yaml")


if __name__ == "__main__":
    main()

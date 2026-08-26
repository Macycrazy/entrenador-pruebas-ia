#!/usr/bin/env python3
"""Genera piezas «normales» y piezas con defecto, para probar la detección de anomalías.

Simula un control de calidad: la pieza correcta es un disco con su textura; las anómalas
tienen una raya, una mancha o una muesca. El modelo se entrena SOLO con las correctas.

    python preparacion/generar_anomalias_sintetico.py --normales 600 --anomalas 60
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

RAIZ = Path(__file__).resolve().parent.parent


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Piezas con y sin defecto")
    p.add_argument("--destino", type=Path, default=RAIZ / "datos_anomalias")
    p.add_argument("--normales", type=int, default=600)
    p.add_argument("--anomalas", type=int, default=60)
    p.add_argument("--tam", type=int, default=128)
    p.add_argument("--semilla", type=int, default=42)
    return p.parse_args()


def pieza(tam: int, aleatorio: random.Random, defecto: bool) -> Image.Image:
    fondo = np.random.default_rng(aleatorio.randint(0, 10**6)).integers(
        26, 38, (tam, tam, 3), dtype=np.uint8)
    imagen = Image.fromarray(fondo)
    d = ImageDraw.Draw(imagen)

    # La pieza: un disco metálico con variación normal de posición y tono
    centro = tam // 2 + aleatorio.randint(-4, 4)
    radio = int(tam * 0.36) + aleatorio.randint(-3, 3)
    tono = aleatorio.randint(140, 175)
    d.ellipse([centro - radio, centro - radio, centro + radio, centro + radio],
              fill=(tono, tono, tono + 8))
    d.ellipse([centro - radio // 3, centro - radio // 3,
               centro + radio // 3, centro + radio // 3],
              outline=(tono - 45, tono - 45, tono - 38), width=2)

    if defecto:
        tipo = aleatorio.choice(["raya", "mancha", "muesca"])
        if tipo == "raya":
            x = centro + aleatorio.randint(-radio // 2, radio // 2)
            d.line([x, centro - radio + 6, x + aleatorio.randint(-8, 8), centro + radio - 6],
                   fill=(60, 55, 50), width=aleatorio.randint(2, 4))
        elif tipo == "mancha":
            x = centro + aleatorio.randint(-radio // 2, radio // 2)
            y = centro + aleatorio.randint(-radio // 2, radio // 2)
            r = aleatorio.randint(5, 11)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(70, 60, 45))
        else:
            ang = aleatorio.random() * 6.28
            x = centro + int(radio * np.cos(ang))
            y = centro + int(radio * np.sin(ang))
            r = aleatorio.randint(7, 13)
            d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v) for v in fondo[0, 0]))

    imagen = imagen.filter(ImageFilter.GaussianBlur(0.4))
    array = np.asarray(imagen).astype(np.float32) + np.random.default_rng(
        aleatorio.randint(0, 10**6)).normal(0, 3, (tam, tam, 3))
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def main() -> None:
    args = argumentos()
    aleatorio = random.Random(args.semilla)
    for nombre, cuantas, defecto in (("normal", args.normales, False),
                                     ("anomalas", args.anomalas, True)):
        carpeta = args.destino / nombre
        carpeta.mkdir(parents=True, exist_ok=True)
        for i in range(cuantas):
            pieza(args.tam, aleatorio, defecto).save(carpeta / f"{i:05d}.jpg", quality=92)
        print(f"{nombre}: {cuantas} imágenes")
    print(f"\nEn {args.destino}")
    print("Ahora:  python entrenar.py --config configs/anomalias.yaml")


if __name__ == "__main__":
    main()

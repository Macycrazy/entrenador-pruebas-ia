#!/usr/bin/env python3
"""Genera audios sintéticos para probar el circuito de sonido sin descargar nada.

Tres clases fáciles de distinguir por su espectro: tono grave, tono agudo y ruido.
Sirve para comprobar que el espectrograma, el entrenamiento y las métricas funcionan.

    python preparacion/generar_audio_sintetico.py --por-clase 60

Para datos reales: Speech Commands (palabras clave), Common Voice (voz en español)
o VoxCeleb (identificación de hablante), todos en Hugging Face.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
CLASES = {"grave": (110, 260), "agudo": (900, 2200), "ruido": None}


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audios sintéticos de prueba")
    p.add_argument("--destino", type=Path, default=RAIZ / "datos_audio")
    p.add_argument("--por-clase", type=int, default=60)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--duracion", type=float, default=3.0)
    p.add_argument("--semilla", type=int, default=42)
    return p.parse_args()


def escribir_wav(ruta: Path, senal: np.ndarray, sr: int) -> None:
    datos = np.clip(senal, -1, 1)
    with wave.open(str(ruta), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((datos * 32767).astype("<i2").tobytes())


def main() -> None:
    args = argumentos()
    generador = np.random.default_rng(args.semilla)
    n = int(args.sr * args.duracion)
    t = np.arange(n) / args.sr

    for clase, rango in CLASES.items():
        carpeta = args.destino / clase
        carpeta.mkdir(parents=True, exist_ok=True)
        for indice in range(args.por_clase):
            if rango is None:
                senal = generador.normal(0, 0.25, n)
            else:
                frecuencia = generador.uniform(*rango)
                senal = 0.5 * np.sin(2 * np.pi * frecuencia * t)
                # armónicos y vibrato para que no sea un tono puro trivial
                senal += 0.2 * np.sin(2 * np.pi * frecuencia * 2 * t)
                senal *= 1 + 0.1 * np.sin(2 * np.pi * generador.uniform(2, 6) * t)
                senal += generador.normal(0, 0.05, n)
            escribir_wav(carpeta / f"{indice:04d}.wav", senal, args.sr)
        print(f"{clase}: {args.por_clase} audios")

    print(f"\nEn {args.destino}")
    print("Ahora:  python entrenar.py --config configs/voz.yaml --set entrenamiento.epocas=3")


if __name__ == "__main__":
    main()

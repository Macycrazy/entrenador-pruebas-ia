#!/usr/bin/env python3
"""Monitor de la GPU: temperatura, consumo y uso, con registro opcional a CSV.

Se ejecuta en otra terminal mientras entrenas:

    python preparacion/vigilar_gpu.py
    python preparacion/vigilar_gpu.py --csv temperatura.csv --cada 5

Referencias para una RTX 5060 Ti (180 W):
  < 80 °C   perfecto
  80-84 °C  normal bajo carga sostenida
  > 85 °C   revisa polvo y flujo de aire; la tarjeta ya estará bajando frecuencias
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from nucleo.vigilante import VigilanteGPU


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor de temperatura y consumo de la GPU")
    p.add_argument("--cada", type=float, default=2.0, help="Segundos entre lecturas")
    p.add_argument("--csv", type=Path, help="Archivo donde registrar el histórico")
    p.add_argument("--limite", type=int, default=85, help="°C a partir de los que avisa")
    return p.parse_args()


def main() -> None:
    args = argumentos()
    vigilante = VigilanteGPU(args.limite, tolerancia=10**9, cada_segundos=0)
    if vigilante.metodo is None:
        raise SystemExit("No se puede leer la GPU (¿hay driver NVIDIA y nvidia-smi?)")

    escritor = None
    if args.csv:
        archivo = args.csv.open("w", newline="")
        escritor = csv.writer(archivo)
        escritor.writerow(["hora", "temp_c", "potencia_w", "uso_pct"])

    print(f"Leyendo por '{vigilante.metodo}' cada {args.cada:g}s. Ctrl-C para salir.\n")
    pico, inicio = 0.0, time.monotonic()
    try:
        while True:
            lectura = vigilante.leer()
            if lectura:
                pico = max(pico, lectura["temp"])
                barra = "■" * int(min(40, lectura["temp"] / 2.5))
                alerta = "  <<< por encima del límite" if lectura["temp"] >= args.limite else ""
                print(f"\r{lectura['temp']:>5.0f} °C {barra:<40} "
                      f"{lectura['potencia']:>6.1f} W  {lectura['uso']:>3.0f}%  "
                      f"(pico {pico:.0f} °C){alerta}   ", end="", flush=True)
                if escritor:
                    escritor.writerow([datetime.now().isoformat(timespec="seconds"),
                                       lectura["temp"], lectura["potencia"], lectura["uso"]])
            time.sleep(args.cada)
    except KeyboardInterrupt:
        minutos = (time.monotonic() - inicio) / 60
        print(f"\n\n{minutos:.1f} min vigilados · pico {pico:.0f} °C")
        if args.csv:
            archivo.close()
            print(f"Registro en {args.csv}")


if __name__ == "__main__":
    main()

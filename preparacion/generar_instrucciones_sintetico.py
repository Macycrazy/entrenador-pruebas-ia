#!/usr/bin/env python3
"""Genera un JSONL de instrucciones sintéticas para probar el ajuste de un LLM.

    python preparacion/generar_instrucciones_sintetico.py --cuantas 400
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

PLANTILLAS = [
    ("Clasifica la solicitud como queja, consulta o solicitud.",
     "El {cosa} lleva {tiempo} sin funcionar y nadie responde.", "queja"),
    ("Clasifica la solicitud como queja, consulta o solicitud.",
     "¿Cuánto tarda el trámite del {cosa}?", "consulta"),
    ("Clasifica la solicitud como queja, consulta o solicitud.",
     "Necesito un nuevo {cosa} para el personal.", "solicitud"),
    ("Extrae el objeto mencionado en el texto.",
     "Ayer entregué el {cosa} en la oficina.", "{cosa}"),
    ("Responde con el plazo mencionado.",
     "El {cosa} estará listo en {tiempo}.", "{tiempo}"),
]
COSAS = ["carnet", "permiso", "expediente", "usuario", "reporte mensual", "equipo asignado"]
TIEMPOS = ["dos días", "una semana", "un mes", "quince días"]


def main() -> None:
    p = argparse.ArgumentParser(description="Instrucciones sintéticas de prueba")
    p.add_argument("--destino", type=Path, default=RAIZ / "datos_llm")
    p.add_argument("--cuantas", type=int, default=400)
    p.add_argument("--semilla", type=int, default=42)
    args = p.parse_args()

    aleatorio = random.Random(args.semilla)
    filas = []
    for _ in range(args.cuantas):
        instruccion, entrada, salida = aleatorio.choice(PLANTILLAS)
        cosa, tiempo = aleatorio.choice(COSAS), aleatorio.choice(TIEMPOS)
        filas.append({
            "instruccion": instruccion,
            "entrada": entrada.format(cosa=cosa, tiempo=tiempo),
            "salida": salida.format(cosa=cosa, tiempo=tiempo),
        })

    aleatorio.shuffle(filas)
    corte = max(1, int(len(filas) * 0.1))
    args.destino.mkdir(parents=True, exist_ok=True)
    for nombre, subconjunto in (("val.jsonl", filas[:corte]), ("train.jsonl", filas[corte:])):
        (args.destino / nombre).write_text(
            "\n".join(json.dumps(f, ensure_ascii=False) for f in subconjunto) + "\n",
            encoding="utf-8")
        print(f"{nombre}: {len(subconjunto)} ejemplos")
    print(f"\nEn {args.destino}\nAhora:  python entrenar.py --config configs/llm.yaml")


if __name__ == "__main__":
    main()

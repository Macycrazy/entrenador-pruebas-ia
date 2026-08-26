#!/usr/bin/env python3
"""Genera un CSV de texto sintético para probar el circuito de NLP sin descargar datos.

Tres intenciones típicas de una mesa de ayuda, construidas combinando plantillas:
queja, consulta y solicitud.

    python preparacion/generar_texto_sintetico.py --por-clase 300
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

PLANTILLAS = {
    "queja": [
        "El {cosa} lleva {tiempo} sin funcionar y nadie responde.",
        "Estoy molesto porque mi {cosa} sigue con el mismo problema desde hace {tiempo}.",
        "Es inaceptable, ya reclamé por el {cosa} y no han hecho nada en {tiempo}.",
        "Pésimo servicio: el {cosa} falla otra vez y perdí {tiempo} esperando.",
    ],
    "consulta": [
        "Buenos días, ¿cómo puedo saber el estado de mi {cosa}?",
        "Quisiera saber cuánto tarda el trámite del {cosa}.",
        "¿Me pueden explicar qué requisitos hacen falta para el {cosa}?",
        "Una pregunta: ¿el {cosa} se puede gestionar en línea?",
    ],
    "solicitud": [
        "Necesito solicitar un nuevo {cosa} para el personal de la oficina.",
        "Por favor, tramiten el {cosa} a la brevedad posible.",
        "Solicito formalmente la renovación del {cosa} antes de {tiempo}.",
        "Requiero que me emitan el {cosa} con los datos actualizados.",
    ],
}
COSAS = ["carnet", "sistema de acceso", "usuario", "reporte mensual", "permiso",
         "equipo asignado", "correo institucional", "expediente"]
TIEMPOS = ["dos días", "una semana", "un mes", "varias horas", "quince días"]


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Texto sintético de prueba")
    p.add_argument("--destino", type=Path, default=RAIZ / "datos_texto")
    p.add_argument("--por-clase", type=int, default=300)
    p.add_argument("--semilla", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = argumentos()
    aleatorio = random.Random(args.semilla)
    filas = []

    for clase, plantillas in PLANTILLAS.items():
        for _ in range(args.por_clase):
            texto = aleatorio.choice(plantillas).format(
                cosa=aleatorio.choice(COSAS), tiempo=aleatorio.choice(TIEMPOS))
            if aleatorio.random() < 0.3:
                texto += " " + aleatorio.choice(
                    ["Gracias.", "Quedo atento.", "Espero respuesta pronto.", ""])
            filas.append({"texto": texto.strip(), "etiqueta": clase})

    aleatorio.shuffle(filas)
    corte = int(len(filas) * 0.15)
    args.destino.mkdir(parents=True, exist_ok=True)

    for nombre, subconjunto in (("val.csv", filas[:corte]), ("train.csv", filas[corte:])):
        with (args.destino / nombre).open("w", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=["texto", "etiqueta"])
            escritor.writeheader()
            escritor.writerows(subconjunto)
        print(f"{nombre}: {len(subconjunto)} filas")

    print(f"\nEn {args.destino}")
    print("Ahora:  python entrenar.py --config configs/texto.yaml")


if __name__ == "__main__":
    main()

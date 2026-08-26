#!/usr/bin/env python3
"""Genera datos tabulares y una serie temporal de prueba, con estructura realista.

    python preparacion/generar_tabular_sintetico.py --filas 3000

Crea dos cosas:
  datos_tabular/datos.csv   personal ficticio, con una columna «ausento» que depende
                            de verdad de las demás (para que haya algo que aprender)
  datos_series/serie.csv    asistencia diaria con tendencia, estacionalidad y ruido
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
GERENCIAS = ["Sistemas", "Recursos Humanos", "Operaciones", "Seguridad", "Administración"]
TURNOS = ["mañana", "tarde", "noche"]


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Datos tabulares y series de prueba")
    p.add_argument("--filas", type=int, default=3000)
    p.add_argument("--dias", type=int, default=730)
    p.add_argument("--semilla", type=int, default=42)
    return p.parse_args()


def tabla(cuantas: int, aleatorio: random.Random) -> list[dict]:
    filas = []
    for i in range(cuantas):
        antiguedad = round(aleatorio.uniform(0, 25), 1)
        distancia = round(aleatorio.uniform(1, 45), 1)
        turno = aleatorio.choice(TURNOS)
        gerencia = aleatorio.choice(GERENCIAS)
        edad = aleatorio.randint(20, 64)
        hijos = aleatorio.randint(0, 4)

        # La probabilidad depende de verdad de las variables: si no, no habría nada
        # que aprender y el modelo no podría superar al azar.
        riesgo = (0.06 + 0.014 * distancia / 10 + 0.10 * (turno == "noche")
                  + 0.03 * hijos - 0.006 * antiguedad + 0.04 * (edad < 26))
        filas.append({
            "id_empleado": f"E{i:05d}",
            "edad": edad, "antiguedad_anios": antiguedad, "hijos": hijos,
            "distancia_km": distancia, "turno": turno, "gerencia": gerencia,
            "horas_extra_mes": round(max(0, aleatorio.gauss(8, 6)), 1),
            "ausento": "si" if aleatorio.random() < min(0.9, max(0.01, riesgo)) else "no",
        })
    return filas


def serie(dias: int, aleatorio: random.Random) -> list[dict]:
    filas, inicio = [], date.today() - timedelta(days=dias)
    for d in range(dias):
        fecha = inicio + timedelta(days=d)
        base = 420 + 0.05 * d                                   # crece poco a poco
        semanal = 40 * math.sin(2 * math.pi * fecha.weekday() / 7)   # baja el fin de semana
        anual = 25 * math.sin(2 * math.pi * d / 365)                 # vacaciones
        fin_de_semana = -180 if fecha.weekday() >= 5 else 0
        valor = base + semanal + anual + fin_de_semana + aleatorio.gauss(0, 12)
        filas.append({"fecha": fecha.isoformat(), "asistencia": round(max(0, valor))})
    return filas


def main() -> None:
    args = argumentos()
    aleatorio = random.Random(args.semilla)

    destino = RAIZ / "datos_tabular"
    destino.mkdir(parents=True, exist_ok=True)
    filas = tabla(args.filas, aleatorio)
    with (destino / "datos.csv").open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    ausentes = sum(1 for f in filas if f["ausento"] == "si")
    print(f"datos_tabular/datos.csv: {len(filas)} filas · {ausentes} ausencias "
          f"({ausentes / len(filas) * 100:.1f} %)")

    destino = RAIZ / "datos_series"
    destino.mkdir(parents=True, exist_ok=True)
    puntos = serie(args.dias, aleatorio)
    with (destino / "serie.csv").open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=["fecha", "asistencia"])
        escritor.writeheader()
        escritor.writerows(puntos)
    print(f"datos_series/serie.csv: {len(puntos)} días de asistencia")

    print("\nAhora:")
    print("  python entrenar.py --config configs/tabular.yaml --set tabular.objetivo=ausento "
          "--set 'tabular.ignorar=[\"id_empleado\"]'")
    print("  python entrenar.py --config configs/series.yaml --set series.columna=asistencia")


if __name__ == "__main__":
    main()

"""Datos por identidad: una carpeta por persona.

    datos_rostros/
        Juan_Perez/  0001.jpg 0002.jpg …
        Ana_Gomez/   0001.jpg …

El split no separa identidades (es identificación de conjunto cerrado): reserva
algunas fotos **de cada persona** para validar. Así se mide si reconoce a alguien
conocido en una foto que no ha visto, que es el caso real de un control de acceso.
"""

from __future__ import annotations

import random
from pathlib import Path

from tareas.imagen_clasificacion.datos import EXTENSIONES, Muestra


def recopilar(cfg) -> tuple[list[Muestra], list[Muestra], list[str]]:
    raiz = Path(cfg.datos.ruta)
    if not raiz.exists():
        raise SystemExit(
            f"No existe {raiz}.\n"
            "Descarga un dataset de identidades:  "
            "python preparacion/descargar_rostros.py --fuente lfw")

    r = cfg.rostros
    aleatorio = random.Random(cfg.semilla)
    identidades, descartadas = [], 0

    for carpeta in sorted(p for p in raiz.iterdir() if p.is_dir()):
        fotos = sorted(f for f in carpeta.rglob("*") if f.suffix.lower() in EXTENSIONES)
        if len(fotos) < max(2, r.min_por_identidad):
            descartadas += 1
            continue
        identidades.append((carpeta.name, fotos))

    if not identidades:
        raise SystemExit(
            f"Ninguna identidad en {raiz} llega a {r.min_por_identidad} fotos.\n"
            "Baja el mínimo con --set rostros.min_por_identidad=2 o usa otro dataset.")

    if r.max_identidades:
        aleatorio.shuffle(identidades)
        identidades = identidades[:r.max_identidades]
    identidades.sort(key=lambda par: par[0])

    clases = [nombre for nombre, _ in identidades]
    train, val = [], []
    for indice, (_, fotos) in enumerate(identidades):
        fotos = list(fotos)
        aleatorio.shuffle(fotos)
        cuantas = min(max(1, r.val_por_identidad), len(fotos) - 1)
        val += [Muestra(f, indice) for f in fotos[:cuantas]]
        train += [Muestra(f, indice) for f in fotos[cuantas:]]

    if cfg.datos.limite:
        aleatorio.shuffle(train)
        train = train[:cfg.datos.limite]

    print(f"{len(clases)} identidades · {len(train)} fotos de entrenamiento · "
          f"{len(val)} de validación"
          + (f" · {descartadas} identidades descartadas por tener pocas fotos"
             if descartadas else ""))
    return train, val, clases

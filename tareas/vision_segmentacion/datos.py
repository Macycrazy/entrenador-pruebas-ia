"""Datos de segmentación: imagen + máscara con el índice de clase en cada píxel.

    datos_segmentacion/
        clases.txt                una clase por línea (la primera es el fondo)
        train/imagenes/001.jpg
        train/mascaras/001.png    PNG en escala de grises: 0=fondo, 1=clase1, …
        val/…

El 255 se reserva para «ignorar» (bordes sin etiquetar), como en Pascal VOC y Cityscapes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORAR = 255


@dataclass
class MuestraSegmentacion:
    imagen: Path
    mascara: Path


def leer_clases(raiz: Path) -> list[str]:
    archivo = raiz / "clases.txt"
    if archivo.exists():
        return [l.strip() for l in archivo.read_text().splitlines() if l.strip()]
    raise SystemExit(f"Falta {archivo} (una clase por línea; la primera es el fondo)")


def recopilar(raiz: Path, split: str) -> list[MuestraSegmentacion]:
    carpeta = raiz / split
    dir_img = carpeta / "imagenes" if (carpeta / "imagenes").exists() else carpeta / "images"
    dir_mask = carpeta / "mascaras" if (carpeta / "mascaras").exists() else carpeta / "masks"
    if not dir_img.exists() or not dir_mask.exists():
        raise SystemExit(f"Faltan {dir_img} o {dir_mask}")

    muestras = []
    for imagen in sorted(p for p in dir_img.iterdir() if p.suffix.lower() in EXTENSIONES):
        mascara = next((dir_mask / f"{imagen.stem}{ext}" for ext in (".png", ".jpg")
                        if (dir_mask / f"{imagen.stem}{ext}").exists()), None)
        if mascara:
            muestras.append(MuestraSegmentacion(imagen, mascara))
    if not muestras:
        raise SystemExit(f"No hay pares imagen/máscara en {carpeta}")
    return muestras


class DatasetSegmentacion(Dataset):
    def __init__(self, muestras, cfg, entrenando: bool):
        self.muestras = muestras
        self.cfg = cfg
        self.entrenando = entrenando
        self.tam = cfg.datos.tam_img

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, indice):
        muestra = self.muestras[indice]
        with Image.open(muestra.imagen) as bruta:
            imagen = bruta.convert("RGB")
        with Image.open(muestra.mascara) as bruta:
            mascara = bruta.convert("L")

        # La máscara SIEMPRE se redimensiona con vecino más cercano: interpolar
        # inventaría índices de clase que no existen.
        imagen = TF.resize(imagen, [self.tam, self.tam])
        mascara = TF.resize(mascara, [self.tam, self.tam],
                            interpolation=TF.InterpolationMode.NEAREST)

        if self.entrenando and random.random() < self.cfg.aumentos.flip:
            imagen, mascara = TF.hflip(imagen), TF.hflip(mascara)
        if self.entrenando and self.cfg.aumentos.color[0]:
            b, c, s, t = self.cfg.aumentos.color
            imagen = TF.adjust_brightness(imagen, 1 + random.uniform(-b, b))
            imagen = TF.adjust_contrast(imagen, 1 + random.uniform(-c, c))

        from tareas.imagen_clasificacion.aumentos import DESV, MEDIA
        x = TF.normalize(TF.to_tensor(imagen), MEDIA, DESV)
        y = torch.from_numpy(np.array(mascara, dtype=np.int64))
        return x, y, {}, {}


def juntar(lote):
    xs, ys, _, _ = zip(*lote)
    return torch.stack(xs), torch.stack(ys), {}, {}


def crear_loaders(cfg, train, val):
    ds_train = DatasetSegmentacion(train, cfg, True)
    ds_val = DatasetSegmentacion(val, cfg, False)
    comunes = dict(num_workers=cfg.datos.workers, collate_fn=juntar,
                   pin_memory=torch.cuda.is_available(),
                   persistent_workers=cfg.datos.workers > 0)
    return (DataLoader(ds_train, batch_size=cfg.datos.batch, shuffle=True,
                       drop_last=len(train) > cfg.datos.batch, **comunes),
            DataLoader(ds_val, batch_size=cfg.datos.batch, shuffle=False, **comunes),
            ds_train, ds_val)

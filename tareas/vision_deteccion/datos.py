"""Datos de detección: formato YOLO (el que exportan Roboflow, LabelImg, CVAT…) o COCO.

    datos_deteccion/
        clases.txt              una clase por línea
        train/imagenes/*.jpg
        train/etiquetas/*.txt   una línea por objeto: clase cx cy w h  (normalizado 0-1)
        val/imagenes/…  val/etiquetas/…

También acepta un COCO JSON por split (train/_annotations.coco.json).

Ojo con la normalización: los detectores de torchvision esperan la imagen en 0-1 y
la normalizan por dentro, así que aquí NO se aplica la media/desviación de ImageNet.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class MuestraDeteccion:
    imagen: Path
    cajas: list = field(default_factory=list)      # [[x1,y1,x2,y2], …] en píxeles
    clases: list = field(default_factory=list)     # índices 1..N (0 es el fondo)


def leer_clases(raiz: Path) -> list[str]:
    archivo = raiz / "clases.txt"
    if archivo.exists():
        return [l.strip() for l in archivo.read_text().splitlines() if l.strip()]
    raise SystemExit(
        f"Falta {archivo} con una clase por línea.\n"
        "Ejemplo:\n  casco\n  chaleco\n  persona")


def recopilar(raiz: Path, split: str, clases: list[str]) -> list[MuestraDeteccion]:
    carpeta = raiz / split
    if not carpeta.exists():
        raise SystemExit(f"No existe {carpeta}")

    coco = next(carpeta.glob("*.coco.json"), None) or next(carpeta.glob("*_annotations.json"), None)
    if coco:
        return _desde_coco(coco, carpeta, clases)
    return _desde_yolo(carpeta, clases)


def _desde_yolo(carpeta: Path, clases: list[str]) -> list[MuestraDeteccion]:
    dir_img = carpeta / "imagenes" if (carpeta / "imagenes").exists() else carpeta / "images"
    dir_lab = carpeta / "etiquetas" if (carpeta / "etiquetas").exists() else carpeta / "labels"
    if not dir_img.exists():
        raise SystemExit(f"No encuentro las imágenes en {carpeta} (ni 'imagenes' ni 'images')")

    muestras = []
    for imagen in sorted(p for p in dir_img.iterdir() if p.suffix.lower() in EXTENSIONES):
        muestra = MuestraDeteccion(imagen)
        etiqueta = dir_lab / f"{imagen.stem}.txt" if dir_lab.exists() else None
        if etiqueta and etiqueta.exists():
            with Image.open(imagen) as img:
                ancho, alto = img.size
            for linea in etiqueta.read_text().splitlines():
                partes = linea.split()
                if len(partes) < 5:
                    continue
                indice, cx, cy, w, h = int(partes[0]), *map(float, partes[1:5])
                x1, y1 = (cx - w / 2) * ancho, (cy - h / 2) * alto
                x2, y2 = (cx + w / 2) * ancho, (cy + h / 2) * alto
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                muestra.cajas.append([x1, y1, x2, y2])
                muestra.clases.append(indice + 1)   # 0 queda reservado al fondo
        muestras.append(muestra)

    if not muestras:
        raise SystemExit(f"No hay imágenes en {dir_img}")
    return muestras


def _desde_coco(ruta: Path, carpeta: Path, clases: list[str]) -> list[MuestraDeteccion]:
    datos = json.loads(ruta.read_text())
    id_a_indice = {c["id"]: i + 1 for i, c in enumerate(
        sorted(datos["categories"], key=lambda c: c["id"]))}
    por_imagen: dict[int, MuestraDeteccion] = {}

    for imagen in datos["images"]:
        ruta_img = carpeta / imagen["file_name"]
        if not ruta_img.exists():
            ruta_img = carpeta / "imagenes" / imagen["file_name"]
        por_imagen[imagen["id"]] = MuestraDeteccion(ruta_img)

    for anotacion in datos.get("annotations", []):
        muestra = por_imagen.get(anotacion["image_id"])
        if muestra is None or anotacion.get("iscrowd"):
            continue
        x, y, w, h = anotacion["bbox"]
        if w < 1 or h < 1:
            continue
        muestra.cajas.append([x, y, x + w, y + h])
        muestra.clases.append(id_a_indice.get(anotacion["category_id"], 1))

    return [m for m in por_imagen.values() if m.imagen.exists()]


class DatasetDeteccion(Dataset):
    def __init__(self, muestras: list[MuestraDeteccion], entrenando: bool, cfg):
        self.muestras = muestras
        self.entrenando = entrenando
        self.cfg = cfg

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, indice):
        muestra = self.muestras[indice]
        with Image.open(muestra.imagen) as bruta:
            imagen = bruta.convert("RGB")
        cajas = torch.tensor(muestra.cajas, dtype=torch.float32).reshape(-1, 4)
        clases = torch.tensor(muestra.clases, dtype=torch.int64)

        if self.entrenando and self.cfg.aumentos.flip and random.random() < self.cfg.aumentos.flip:
            imagen = TF.hflip(imagen)
            if cajas.numel():
                ancho = imagen.width
                cajas = cajas.clone()
                cajas[:, [0, 2]] = ancho - cajas[:, [2, 0]]

        x = TF.to_tensor(imagen)      # 0-1, sin normalizar: el detector lo hace por dentro
        return x, {"boxes": cajas, "labels": clases,
                   "image_id": torch.tensor(indice), "ruta": str(muestra.imagen)}


def juntar(lote):
    """Las imágenes tienen tamaños distintos y cada una un número distinto de objetos."""
    return list(zip(*lote))


def crear_loaders(cfg, train, val):
    ds_train = DatasetDeteccion(train, True, cfg)
    ds_val = DatasetDeteccion(val, False, cfg)
    comunes = dict(num_workers=cfg.datos.workers, collate_fn=juntar,
                   pin_memory=torch.cuda.is_available(),
                   persistent_workers=cfg.datos.workers > 0)
    return (DataLoader(ds_train, batch_size=cfg.datos.batch, shuffle=True, **comunes),
            DataLoader(ds_val, batch_size=cfg.datos.batch, shuffle=False, **comunes),
            ds_train, ds_val)

"""Pares de imagen pequeña y grande, generados al vuelo desde cualquier carpeta de fotos.

No hace falta un dataset especial: se coge un recorte de la foto original como objetivo
y se degrada (reducir, desenfocar, comprimir) para fabricar la entrada. Así el modelo
aprende exactamente a deshacer el destrozo que quieres deshacer.
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def recopilar(cfg) -> tuple[list[Path], list[Path]]:
    raiz = Path(cfg.datos.ruta)
    if not raiz.exists():
        raise SystemExit(f"No existe {raiz}. Vale cualquier carpeta con fotos, "
                         "por ejemplo datos/train/hombre")
    imagenes = [p for p in raiz.rglob("*") if p.suffix.lower() in EXTENSIONES]
    if not imagenes:
        raise SystemExit(f"No hay imágenes en {raiz}")

    aleatorio = random.Random(cfg.semilla)
    aleatorio.shuffle(imagenes)
    if cfg.datos.limite:
        imagenes = imagenes[:cfg.datos.limite]
    corte = max(1, int(len(imagenes) * cfg.datos.val_proporcion))
    return imagenes[corte:], imagenes[:corte]


def degradar(imagen: Image.Image, escala: int, intensidad: float,
             aleatorio: random.Random) -> Image.Image:
    """Fabrica la versión 'mala': reducir + desenfocar + comprimir, como una cámara pobre."""
    ancho, alto = imagen.size
    pequena = imagen
    if intensidad and aleatorio.random() < intensidad:
        pequena = pequena.filter(ImageFilter.GaussianBlur(aleatorio.uniform(0.3, 1.2)))
    pequena = pequena.resize((max(1, ancho // escala), max(1, alto // escala)), Image.BICUBIC)
    if intensidad and aleatorio.random() < intensidad:
        buffer = io.BytesIO()
        pequena.save(buffer, format="JPEG", quality=aleatorio.randint(45, 85))
        buffer.seek(0)
        pequena = Image.open(buffer).convert("RGB")
    return pequena


class DatasetSuperResolucion(Dataset):
    def __init__(self, imagenes: list[Path], cfg, entrenando: bool):
        self.imagenes = imagenes
        self.cfg = cfg
        self.entrenando = entrenando
        self.escala = cfg.superresolucion.escala
        self.parche = cfg.superresolucion.tam_parche

    def __len__(self):
        return len(self.imagenes)

    def __getitem__(self, indice):
        aleatorio = random.Random(indice if not self.entrenando else None)
        with Image.open(self.imagenes[indice]) as bruta:
            imagen = bruta.convert("RGB")

        lado = self.parche
        if imagen.width < lado or imagen.height < lado:
            imagen = TF.resize(imagen, [max(lado, imagen.height), max(lado, imagen.width)])
        # El recorte se alinea a la escala para que las dos versiones encajen exactas
        x = aleatorio.randint(0, imagen.width - lado) if self.entrenando else \
            (imagen.width - lado) // 2
        y = aleatorio.randint(0, imagen.height - lado) if self.entrenando else \
            (imagen.height - lado) // 2
        alta = imagen.crop((x, y, x + lado, y + lado))
        if self.entrenando and aleatorio.random() < 0.5:
            alta = TF.hflip(alta)

        baja = degradar(alta, self.escala, self.cfg.superresolucion.degradacion, aleatorio)
        return TF.to_tensor(baja), TF.to_tensor(alta), {}, {}


def juntar(lote):
    bajas, altas, _, _ = zip(*lote)
    return torch.stack(bajas), torch.stack(altas), {}, {}


def crear_loaders(cfg, train, val):
    comunes = dict(num_workers=cfg.datos.workers, collate_fn=juntar,
                   pin_memory=torch.cuda.is_available(),
                   persistent_workers=cfg.datos.workers > 0)
    return (DataLoader(DatasetSuperResolucion(train, cfg, True), batch_size=cfg.datos.batch,
                       shuffle=True, drop_last=len(train) > cfg.datos.batch, **comunes),
            DataLoader(DatasetSuperResolucion(val, cfg, False), batch_size=cfg.datos.batch,
                       shuffle=False, **comunes))

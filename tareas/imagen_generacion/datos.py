"""Imágenes con su descripción, para enseñarle un sujeto nuevo al modelo.

    datos_generacion/
        001.jpg  001.txt     (el .txt es opcional: si falta se usa generacion.instancia)
        002.jpg  …

Con 15-25 fotos variadas de la misma persona u objeto basta. La frase de instancia debe
llevar una palabra rara ("sks", "zwx") que el modelo no asocie a nada: es la etiqueta con
la que luego lo invocas al generar.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def recopilar(cfg):
    raiz = Path(cfg.datos.ruta)
    if not raiz.exists():
        raise SystemExit(
            f"No existe {raiz}. Pon ahí entre 15 y 25 fotos del sujeto que quieres enseñarle.")
    imagenes = sorted(p for p in raiz.rglob("*") if p.suffix.lower() in EXTENSIONES)
    if not imagenes:
        raise SystemExit(f"No hay imágenes en {raiz}")

    muestras = []
    for imagen in imagenes:
        texto = imagen.with_suffix(".txt")
        muestras.append((imagen, texto.read_text(encoding="utf-8").strip()
                         if texto.exists() else cfg.generacion.instancia))

    aleatorio = random.Random(cfg.semilla)
    aleatorio.shuffle(muestras)
    if cfg.datos.limite:
        muestras = muestras[:cfg.datos.limite]
    corte = max(1, int(len(muestras) * cfg.datos.val_proporcion))
    return muestras[corte:], muestras[:corte]


class DatasetGeneracion(Dataset):
    def __init__(self, muestras, tokenizador, cfg, entrenando: bool):
        self.muestras = muestras
        self.tokenizador = tokenizador
        self.cfg = cfg
        self.entrenando = entrenando

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, indice):
        ruta, texto = self.muestras[indice]
        with Image.open(ruta) as bruta:
            imagen = bruta.convert("RGB")

        lado = self.cfg.generacion.resolucion
        imagen = TF.center_crop(TF.resize(imagen, lado), [lado, lado])
        if self.entrenando and random.random() < 0.5:
            imagen = TF.hflip(imagen)
        # El VAE espera valores en [-1, 1], no en [0, 1]
        pixeles = TF.normalize(TF.to_tensor(imagen), [0.5] * 3, [0.5] * 3)

        ids = self.tokenizador(texto, padding="max_length", truncation=True,
                               max_length=self.tokenizador.model_max_length,
                               return_tensors="pt").input_ids[0]
        return pixeles, ids


def juntar(lote):
    pixeles, ids = zip(*lote)
    return torch.stack(pixeles), torch.stack(ids), {}, {}


def crear_loaders(cfg, train, val, tokenizador):
    comunes = dict(num_workers=cfg.datos.workers, collate_fn=juntar,
                   persistent_workers=cfg.datos.workers > 0)
    return (DataLoader(DatasetGeneracion(train, tokenizador, cfg, True),
                       batch_size=cfg.datos.batch, shuffle=True, **comunes),
            DataLoader(DatasetGeneracion(val, tokenizador, cfg, False),
                       batch_size=cfg.datos.batch, shuffle=False, **comunes))

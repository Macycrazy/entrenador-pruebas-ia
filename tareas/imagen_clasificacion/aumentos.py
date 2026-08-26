"""Aumentaciones de imagen: políticas automáticas + degradación realista de webcam."""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

MEDIA = (0.485, 0.456, 0.406)
DESV = (0.229, 0.224, 0.225)

POLITICAS = ("ninguna", "basica", "randaugment", "trivialaugment", "autoaugment", "augmix")


class DegradacionWebcam:
    """Ensucia la imagen como lo haría una webcam mala.

    Los datasets de rostros son fotos limpias; la cámara real da desenfoque, ruido,
    compresión y poca resolución. Entrenar con esto es lo que más cierra esa brecha.
    """

    def __init__(self, intensidad: float = 0.5):
        self.p = max(0.0, min(1.0, intensidad))

    def __call__(self, imagen: Image.Image) -> Image.Image:
        if self.p == 0 or random.random() > self.p:
            return imagen

        if random.random() < 0.5:                      # desenfoque (foco o movimiento)
            imagen = imagen.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.8)))

        if random.random() < 0.5:                      # sensor de baja resolución
            ancho, alto = imagen.size
            escala = random.uniform(0.35, 0.8)
            pequena = imagen.resize((max(8, int(ancho * escala)), max(8, int(alto * escala))),
                                    Image.BILINEAR)
            imagen = pequena.resize((ancho, alto), Image.BILINEAR)

        if random.random() < 0.6:                      # compresión JPEG agresiva
            buffer = io.BytesIO()
            imagen.convert("RGB").save(buffer, format="JPEG",
                                       quality=random.randint(25, 70))
            buffer.seek(0)
            imagen = Image.open(buffer).convert("RGB")

        if random.random() < 0.4:                      # ruido del sensor con poca luz
            array = np.asarray(imagen).astype(np.float32)
            array += np.random.normal(0, random.uniform(3, 12), array.shape)
            imagen = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

        return imagen


def _politica(cfg):
    a = cfg.aumentos
    nombre = a.politica.lower()
    if nombre == "randaugment":
        return [transforms.RandAugment(num_ops=a.randaugment_n, magnitude=a.randaugment_m)]
    if nombre == "trivialaugment":
        return [transforms.TrivialAugmentWide()]
    if nombre == "autoaugment":
        return [transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)]
    if nombre == "augmix":
        return [transforms.AugMix()]
    if nombre == "basica":
        b, c, s, t = a.color
        return [
            transforms.RandomApply([transforms.RandomRotation(a.rotacion)], p=0.5),
            transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=t),
            transforms.RandomGrayscale(p=a.grises),
        ]
    if nombre == "ninguna":
        return []
    raise SystemExit(f"Política de aumentos '{a.politica}' desconocida. "
                     f"Opciones: {', '.join(POLITICAS)}")


def entrenamiento(cfg, tam_img: int | None = None) -> transforms.Compose:
    a = cfg.aumentos
    tam = tam_img or cfg.datos.tam_img
    pasos = [transforms.RandomResizedCrop(tam, scale=tuple(a.recorte_escala), ratio=(0.8, 1.25))]
    if a.flip:
        pasos.append(transforms.RandomHorizontalFlip(a.flip))
    if a.webcam:
        pasos.append(DegradacionWebcam(a.webcam))
    pasos += _politica(cfg)
    pasos += [transforms.ToTensor(), transforms.Normalize(MEDIA, DESV)]
    if a.borrado:
        pasos.append(transforms.RandomErasing(p=a.borrado, scale=(0.02, 0.15)))
    return transforms.Compose(pasos)


def validacion(cfg, tam_img: int | None = None) -> transforms.Compose:
    tam = tam_img or cfg.datos.tam_img
    return transforms.Compose([
        transforms.Resize(int(tam * 1.14)),
        transforms.CenterCrop(tam),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESV),
    ])

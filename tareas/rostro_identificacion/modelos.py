"""Modelo de reconocimiento facial: backbone → embedding normalizado → cabeza con margen.

La diferencia con un clasificador normal es que lo que se usa después **no** son los
logits sino el embedding: dos fotos de la misma persona deben caer cerca en la esfera
unidad. Por eso la cabeza tiene margen (ArcFace/CosFace) y el cuello normaliza.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nucleo.perdidas import CabezaMargen
from tareas.imagen_clasificacion.modelos import _rostro, _timm, _torchvision


class ModeloRostro(nn.Module):
    def __init__(self, backbone: nn.Module, caracteristicas: int, identidades: int, cfg):
        super().__init__()
        self.backbone = backbone
        dim = cfg.rostros.dim_embedding or caracteristicas
        if cfg.rostros.dim_embedding:
            # Cuello estándar de reconocimiento facial (BN → dropout → lineal → BN):
            # estabiliza mucho el entrenamiento con margen.
            self.cuello = nn.Sequential(
                nn.BatchNorm1d(caracteristicas),
                nn.Dropout(cfg.modelo.dropout),
                nn.Linear(caracteristicas, dim),
                nn.BatchNorm1d(dim),
            )
        else:
            # dim_embedding=0: se usa la salida del backbone tal cual. Es lo que hay
            # que hacer con un backbone ya entrenado con caras (rostro:facenet) si no
            # se va a reentrenar: un cuello sin entrenar destrozaría sus embeddings.
            self.cuello = nn.Identity()
        tipo = cfg.modelo.cabeza.lower()
        self.cabeza = CabezaMargen(dim, identidades, cfg.perdida.arcface_margen,
                                   cfg.perdida.arcface_escala,
                                   tipo if tipo in ("arcface", "cosface") else "arcface")
        self.dim_embedding = dim

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Vector unitario que representa la cara. Es lo que se compara y se almacena."""
        return F.normalize(self.cuello(self.backbone(x)))

    def forward(self, x, objetivo=None, solo_logits: bool = False):
        emb = self.embeddings(x)
        logits = self.cabeza(emb, objetivo)
        if solo_logits:
            return logits
        return {"principal": logits, "rasgos": emb}


def crear_modelo(cfg, identidades: int) -> ModeloRostro:
    nombre = cfg.modelo.arquitectura
    if nombre.startswith("timm:"):
        backbone, caracteristicas = _timm(nombre[5:], cfg)
    elif nombre.startswith("rostro:"):
        backbone, caracteristicas = _rostro(nombre[7:], cfg)
    else:
        backbone, caracteristicas = _torchvision(nombre, cfg)
    return ModeloRostro(backbone, caracteristicas, identidades, cfg)

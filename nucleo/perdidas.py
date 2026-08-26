"""Funciones de pérdida, cabezas con margen y mezcla de muestras (mixup/cutmix)."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def crear_perdida(cfg, pesos: torch.Tensor | None = None) -> nn.Module:
    p = cfg.perdida
    nombre = p.nombre.lower()
    if nombre == "ce":
        return nn.CrossEntropyLoss(weight=pesos, label_smoothing=p.suavizado)
    if nombre == "focal":
        return PerdidaFocal(gamma=p.focal_gamma, pesos=pesos, suavizado=p.suavizado)
    if nombre == "bce":
        return PerdidaBCE(pesos=pesos, suavizado=p.suavizado)
    if nombre in ("arcface", "cosface"):
        # El margen vive en la cabeza; aquí solo se aplica la entropía cruzada.
        return nn.CrossEntropyLoss(weight=pesos, label_smoothing=p.suavizado)
    raise SystemExit(f"Pérdida '{p.nombre}' desconocida")


class PerdidaFocal(nn.Module):
    """Baja el peso de los ejemplos fáciles para que el modelo mire los difíciles.

    Útil con clases desbalanceadas o cuando un puñado de casos raros son los que fallan.
    """

    def __init__(self, gamma: float = 2.0, pesos: torch.Tensor | None = None,
                 suavizado: float = 0.0):
        super().__init__()
        self.gamma, self.suavizado = gamma, suavizado
        self.register_buffer("pesos", pesos if pesos is not None else torch.tensor([]))

    def forward(self, logits, objetivo):
        pesos = self.pesos if self.pesos.numel() else None
        ce = F.cross_entropy(logits, objetivo, weight=pesos,
                             label_smoothing=self.suavizado, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class PerdidaBCE(nn.Module):
    """Binaria sobre logits de 2 clases: a veces calibra mejor que la softmax."""

    def __init__(self, pesos: torch.Tensor | None = None, suavizado: float = 0.0):
        super().__init__()
        self.suavizado = suavizado
        self.register_buffer("pesos", pesos if pesos is not None else torch.tensor([]))

    def forward(self, logits, objetivo):
        objetivo_1h = F.one_hot(objetivo, logits.size(1)).float()
        if self.suavizado:
            objetivo_1h = objetivo_1h * (1 - self.suavizado) + self.suavizado / logits.size(1)
        pesos = self.pesos if self.pesos.numel() else None
        peso_muestra = pesos[objetivo].unsqueeze(1) if pesos is not None else None
        return F.binary_cross_entropy_with_logits(logits, objetivo_1h, weight=peso_muestra)


class PerdidaDestilacion(nn.Module):
    """Combina la etiqueta real con las probabilidades suaves de un modelo profesor."""

    def __init__(self, base: nn.Module, temperatura: float = 4.0, alfa: float = 0.5):
        super().__init__()
        self.base, self.T, self.alfa = base, temperatura, alfa

    def forward(self, logits, objetivo, logits_profesor=None):
        dura = self.base(logits, objetivo)
        if logits_profesor is None:
            return dura
        blanda = F.kl_div(
            F.log_softmax(logits / self.T, dim=1),
            F.softmax(logits_profesor / self.T, dim=1),
            reduction="batchmean",
        ) * self.T ** 2
        return (1 - self.alfa) * dura + self.alfa * blanda


class CabezaMargen(nn.Module):
    """Cabeza ArcFace/CosFace: separa las clases en la esfera, no con un plano.

    Es lo que usan los sistemas de reconocimiento facial; también ayuda en
    clasificación fina con pocas muestras por clase.
    """

    def __init__(self, entradas: int, clases: int, margen: float = 0.5,
                 escala: float = 30.0, tipo: str = "arcface"):
        super().__init__()
        self.peso = nn.Parameter(torch.empty(clases, entradas))
        nn.init.xavier_normal_(self.peso)
        self.margen, self.escala, self.tipo = margen, escala, tipo

    def forward(self, caracteristicas, objetivo=None):
        coseno = F.linear(F.normalize(caracteristicas), F.normalize(self.peso)).clamp(-1 + 1e-7, 1 - 1e-7)
        if objetivo is None:          # inferencia: sin margen
            return coseno * self.escala
        if self.tipo == "cosface":
            margen = torch.zeros_like(coseno).scatter_(1, objetivo.view(-1, 1), self.margen)
            return (coseno - margen) * self.escala
        theta = torch.acos(coseno)
        margen = torch.zeros_like(theta).scatter_(1, objetivo.view(-1, 1), self.margen)
        return torch.cos(theta + margen) * self.escala


# ---------------------------------------------------------------- mixup / cutmix

def mezclar_lote(x: torch.Tensor, y: torch.Tensor, cfg):
    """Aplica mixup o cutmix. Devuelve (x, y_a, y_b, lam); lam=1 significa 'sin mezcla'."""
    a = cfg.aumentos
    usar_mixup, usar_cutmix = a.mixup > 0, a.cutmix > 0
    if not (usar_mixup or usar_cutmix) or np.random.rand() > a.mixcut_prob:
        return x, y, y, 1.0

    if usar_mixup and (not usar_cutmix or np.random.rand() < 0.5):
        lam = float(np.random.beta(a.mixup, a.mixup))
        indices = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1 - lam) * x[indices], y, y[indices], lam

    lam = float(np.random.beta(a.cutmix, a.cutmix))
    indices = torch.randperm(x.size(0), device=x.device)
    alto, ancho = x.shape[-2:]
    ratio = math.sqrt(1 - lam)
    ch, cw = int(alto * ratio), int(ancho * ratio)
    cy, cx = np.random.randint(alto), np.random.randint(ancho)
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, alto)
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, ancho)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[indices, :, y1:y2, x1:x2]
    lam = 1 - ((y2 - y1) * (x2 - x1) / (alto * ancho))
    return x, y, y[indices], lam


def perdida_mezclada(criterio, salida, y_a, y_b, lam: float):
    if lam == 1.0:
        return criterio(salida, y_a)
    return lam * criterio(salida, y_a) + (1 - lam) * criterio(salida, y_b)


def pesos_de_clase(conteo: list[int], dispositivo="cpu") -> torch.Tensor:
    """Frecuencia inversa normalizada, para pasar a la pérdida."""
    total = sum(conteo)
    pesos = [total / (len(conteo) * max(1, n)) for n in conteo]
    return torch.tensor(pesos, dtype=torch.float32, device=dispositivo)

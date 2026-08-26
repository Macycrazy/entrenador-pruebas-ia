"""Red de super-resolución: reconstruye detalle a partir de una imagen pequeña.

Es una arquitectura pequeña al estilo EDSR: una convolución de entrada, unos bloques
residuales y una subida de resolución con PixelShuffle. Cabe entrenarla desde cero
—no hace falta ningún modelo preentrenado— y con pocas horas ya se nota.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BloqueResidual(nn.Module):
    def __init__(self, canales: int, escala_residual: float = 0.1):
        super().__init__()
        self.cuerpo = nn.Sequential(
            nn.Conv2d(canales, canales, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(canales, canales, 3, padding=1),
        )
        # Escalar el residuo estabiliza el entrenamiento con muchos bloques (truco de EDSR)
        self.escala = escala_residual

    def forward(self, x):
        return x + self.cuerpo(x) * self.escala


class Subida(nn.Sequential):
    """PixelShuffle: la red aprende canales extra y se reordenan como píxeles."""

    def __init__(self, canales: int, escala: int):
        capas = []
        if escala in (2, 4, 8):
            for _ in range(int(escala).bit_length() - 1):
                capas += [nn.Conv2d(canales, canales * 4, 3, padding=1), nn.PixelShuffle(2)]
        elif escala == 3:
            capas += [nn.Conv2d(canales, canales * 9, 3, padding=1), nn.PixelShuffle(3)]
        else:
            raise SystemExit(f"Escala {escala} no soportada (usa 2, 3, 4 u 8)")
        super().__init__(*capas)


class SuperResolucion(nn.Module):
    def __init__(self, escala: int = 4, canales: int = 64, bloques: int = 8):
        super().__init__()
        self.escala = escala
        self.entrada = nn.Conv2d(3, canales, 3, padding=1)
        self.cuerpo = nn.Sequential(*[BloqueResidual(canales) for _ in range(bloques)],
                                    nn.Conv2d(canales, canales, 3, padding=1))
        self.subida = Subida(canales, escala)
        self.salida = nn.Conv2d(canales, 3, 3, padding=1)

    def forward(self, x):
        rasgos = self.entrada(x)
        rasgos = rasgos + self.cuerpo(rasgos)
        # La red solo aprende el detalle que falta; la base la pone un redimensionado simple
        base = nn.functional.interpolate(x, scale_factor=self.escala, mode="bicubic",
                                         align_corners=False)
        return (self.salida(self.subida(rasgos)) + base).clamp(0, 1)


def crear_modelo(cfg) -> SuperResolucion:
    s = cfg.superresolucion
    return SuperResolucion(s.escala, s.canales, s.bloques)

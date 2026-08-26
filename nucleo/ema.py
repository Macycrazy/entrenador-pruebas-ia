"""Promediado de pesos: EMA (media móvil exponencial) y SWA (promedio de la cola)."""

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
import torch.nn as nn


class EMA:
    """Mantiene una copia suavizada de los pesos. Casi siempre da algo de precisión gratis.

    El decaimiento arranca bajo y sube hasta `decaimiento`, para que la copia no se
    quede anclada a la inicialización durante las primeras iteraciones.
    """

    def __init__(self, modelo: nn.Module, decaimiento: float = 0.999, calentamiento: int = 2000):
        self.decaimiento, self.calentamiento, self.pasos = decaimiento, calentamiento, 0
        self.sombra = copy.deepcopy(_sin_compilar(modelo)).eval()
        for p in self.sombra.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def actualizar(self, modelo: nn.Module) -> None:
        self.pasos += 1
        d = self.decaimiento * (1 - torch.exp(torch.tensor(-self.pasos / self.calentamiento))).item()
        origen = _sin_compilar(modelo).state_dict()
        for clave, valor in self.sombra.state_dict().items():
            nuevo = origen[clave]
            if valor.dtype.is_floating_point:
                valor.mul_(d).add_(nuevo.detach(), alpha=1 - d)
            else:
                valor.copy_(nuevo)

    @contextmanager
    def aplicado(self, modelo: nn.Module):
        """Intercambia temporalmente los pesos del modelo por los de la EMA."""
        base = _sin_compilar(modelo)
        respaldo = copy.deepcopy(base.state_dict())
        base.load_state_dict(self.sombra.state_dict())
        try:
            yield modelo
        finally:
            base.load_state_dict(respaldo)

    def state_dict(self):
        return {"sombra": self.sombra.state_dict(), "pasos": self.pasos}

    def load_state_dict(self, estado):
        self.sombra.load_state_dict(estado["sombra"])
        self.pasos = estado.get("pasos", 0)


class SWA:
    """Promedio simple de los pesos de las últimas épocas (Stochastic Weight Averaging)."""

    def __init__(self, modelo: nn.Module):
        self.media = copy.deepcopy(_sin_compilar(modelo)).eval()
        for p in self.media.parameters():
            p.requires_grad_(False)
        self.n = 0

    @torch.no_grad()
    def actualizar(self, modelo: nn.Module) -> None:
        self.n += 1
        origen = _sin_compilar(modelo).state_dict()
        for clave, valor in self.media.state_dict().items():
            nuevo = origen[clave]
            if valor.dtype.is_floating_point:
                valor.add_((nuevo.detach() - valor) / self.n)
            else:
                valor.copy_(nuevo)

    @torch.no_grad()
    def recalcular_bn(self, loader, dispositivo) -> None:
        """SWA invalida las estadísticas de BatchNorm: hay que recorrer los datos otra vez."""
        modulos_bn = [m for m in self.media.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        if not modulos_bn:
            return
        momentos = [m.momentum for m in modulos_bn]
        for m in modulos_bn:
            m.reset_running_stats()
            m.momentum = None
        self.media.train()
        for lote in loader:
            x = lote[0] if isinstance(lote, (list, tuple)) else lote
            self.media(x.to(dispositivo))
        self.media.eval()
        for m, momento in zip(modulos_bn, momentos):
            m.momentum = momento


def _sin_compilar(modelo: nn.Module) -> nn.Module:
    return getattr(modelo, "_orig_mod", modelo)

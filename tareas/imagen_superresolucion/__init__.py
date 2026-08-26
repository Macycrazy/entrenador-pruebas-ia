"""Tarea: super-resolución — recuperar detalle de una foto pequeña o borrosa.

Útil de verdad aquí: rescatar fotos de carnet viejas o tomadas con una cámara mala.
Se entrena desde cero, sin modelo preentrenado, porque el problema es local (texturas
y bordes) y no necesita el conocimiento del mundo que aporta ImageNet.

La métrica es **PSNR** en decibelios: cuánto se parece la reconstrucción al original.
Cada 3 dB significa la mitad de error. Un redimensionado normal ronda los 26-28 dB;
un modelo entrenado sube varios puntos por encima de eso.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from nucleo.tarea import InfoDatos, Paso, Tarea, registrar

from . import datos as datos_mod, modelos


@registrar("imagen_superresolucion")
class TareaSuperResolucion(Tarea):

    def datos(self):
        train, val = datos_mod.recopilar(self.cfg)
        loader_train, loader_val = datos_mod.crear_loaders(self.cfg, train, val)
        s = self.cfg.superresolucion
        print(f"super-resolución ×{s.escala} · parches de {s.tam_parche}px · "
              f"degradación {s.degradacion}")
        return loader_train, loader_val, InfoDatos(
            clases=["imagen"], conteo=[len(train)], n_train=len(train), n_val=len(val))

    def modelo(self, info: InfoDatos):
        modelo = modelos.crear_modelo(self.cfg)
        parametros = sum(p.numel() for p in modelo.parameters())
        print(f"red de {parametros / 1e6:.2f} M parámetros")
        return modelo

    def criterio(self, info: InfoDatos, dispositivo):
        # L1 da imágenes más nítidas que L2, que tiende a promediar y emborronar
        return nn.L1Loss().to(dispositivo)

    def paso(self, modelo, lote, criterio, dispositivo, entrenando: bool,
             espejo: bool = False) -> Paso:
        baja, alta, _, _ = lote
        baja = baja.to(dispositivo, non_blocking=True)
        alta = alta.to(dispositivo, non_blocking=True)
        reconstruida = modelo(baja)
        cuantas = torch.zeros(baja.size(0))
        return Paso(perdida=criterio(reconstruida, alta), logits=cuantas, objetivos=cuantas,
                    datos_extra={"reconstruida": reconstruida.detach(), "alta": alta,
                                 "baja": baja})

    def evaluador(self, info: InfoDatos):
        return EvaluadorSuperResolucion(self.cfg)

    def descripcion(self) -> str:
        return f"super-resolución ×{self.cfg.superresolucion.escala}"

    def exportar_extra(self) -> dict:
        return {"escala": self.cfg.superresolucion.escala,
                "canales": self.cfg.superresolucion.canales,
                "bloques": self.cfg.superresolucion.bloques,
                "arquitectura": f"edsr_lite_x{self.cfg.superresolucion.escala}"}


class EvaluadorSuperResolucion:
    """PSNR del modelo y, para comparar, el del redimensionado bicúbico de toda la vida."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.metrica_objetivo = cfg.entrenamiento.metrica_objetivo
        self.reiniciar()

    def reiniciar(self) -> None:
        self.psnr_modelo, self.psnr_base, self.n = 0.0, 0.0, 0
        self._perdida = 0.0

    def actualizar(self, logits=None, objetivos=None, perdida=None, subgrupos=None,
                   datos_extra=None) -> None:
        if perdida is not None:
            self._perdida += perdida
        if not datos_extra:
            return
        reconstruida, alta, baja = (datos_extra["reconstruida"], datos_extra["alta"],
                                    datos_extra["baja"])
        base = nn.functional.interpolate(baja, size=alta.shape[-2:], mode="bicubic",
                                         align_corners=False).clamp(0, 1)
        self.psnr_modelo += _psnr(reconstruida, alta) * alta.size(0)
        self.psnr_base += _psnr(base, alta) * alta.size(0)
        self.n += alta.size(0)

    def resumen(self, calibrar: bool = False, curvas: bool = True) -> dict:
        if not self.n:
            return {}
        psnr = self.psnr_modelo / self.n
        base = self.psnr_base / self.n
        return {
            # 'acc' es lo que usa el núcleo para elegir el mejor checkpoint: se normaliza
            # el PSNR a 0-1 dividiendo por 50 dB, que nadie alcanza en la práctica.
            "acc": psnr / 50, "acc_balanceada": psnr / 50,
            "psnr": psnr, "psnr_bicubico": base, "mejora_db": psnr - base,
            "perdida": self._perdida / max(1, len(str(self.n))), "n": self.n,
            "texto": (f"PSNR {psnr:.2f} dB · redimensionado normal {base:.2f} dB · "
                      f"mejora {psnr - base:+.2f} dB"),
        }


def _psnr(prediccion: torch.Tensor, objetivo: torch.Tensor) -> float:
    error = torch.mean((prediccion.float() - objetivo.float()) ** 2).item()
    return 100.0 if error <= 1e-10 else 10 * math.log10(1.0 / error)

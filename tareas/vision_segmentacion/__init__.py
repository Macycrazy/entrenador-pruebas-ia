"""Tarea: segmentación semántica (qué clase es cada píxel).

Sirve para recortar el fondo de una foto de carnet, medir superficies, aislar el uniforme
o cualquier cosa donde una caja no basta. La métrica es **mIoU**: cuánto se solapa lo
predicho con lo real, promediado por clase — mucho más exigente que el acierto por píxel,
que se dispara solo con acertar el fondo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import segmentation

from nucleo.tarea import InfoDatos, Paso, Tarea, registrar

from . import datos as datos_mod

MODELOS = {
    "deeplabv3_resnet50": "deeplab",
    "deeplabv3_resnet101": "deeplab",
    "deeplabv3_mobilenet_v3_large": "deeplab",
    "fcn_resnet50": "fcn",
    "fcn_resnet101": "fcn",
    "lraspp_mobilenet_v3_large": "lraspp",     # el más rápido, para tiempo real
}
PESO_AUXILIAR = 0.4      # las cabezas auxiliares estabilizan el entrenamiento


@registrar("vision_segmentacion")
class TareaVisionSegmentacion(Tarea):

    def datos(self):
        raiz = Path(self.cfg.datos.ruta)
        clases = datos_mod.leer_clases(raiz)
        train = datos_mod.recopilar(raiz, "train")
        val = datos_mod.recopilar(raiz, "val")
        self.clases = clases
        if self.cfg.datos.limite:
            train = train[:self.cfg.datos.limite]
            val = val[:max(1, self.cfg.datos.limite // 5)]

        loader_train, loader_val, self.ds_train, self.ds_val = datos_mod.crear_loaders(
            self.cfg, train, val)
        return loader_train, loader_val, InfoDatos(
            clases=clases, conteo=[0] * len(clases), n_train=len(train), n_val=len(val))

    def modelo(self, info: InfoDatos):
        nombre = self.cfg.modelo.arquitectura
        if nombre not in MODELOS:
            raise SystemExit(f"Modelo de segmentación '{nombre}' desconocido. "
                             f"Opciones: {', '.join(sorted(MODELOS))}")
        pesos = "DEFAULT" if self.cfg.modelo.preentrenado else None
        aux = MODELOS[nombre] != "lraspp"
        modelo = getattr(segmentation, nombre)(
            weights=pesos, **({"aux_loss": True} if aux else {}))
        _cambiar_cabezas(modelo, len(info.clases))
        return modelo

    def criterio(self, info: InfoDatos, dispositivo):
        return nn.CrossEntropyLoss(ignore_index=datos_mod.IGNORAR,
                                   label_smoothing=self.cfg.perdida.suavizado).to(dispositivo)

    def paso(self, modelo, lote, criterio, dispositivo, entrenando: bool,
             espejo: bool = False) -> Paso:
        x, y, _, _ = lote
        x = x.to(dispositivo, non_blocking=True)
        y = y.to(dispositivo, non_blocking=True)
        salida = modelo(x)
        logits = salida["out"]

        perdida = criterio(logits, y)
        if entrenando and "aux" in salida:
            perdida = perdida + PESO_AUXILIAR * criterio(salida["aux"], y)

        return Paso(perdida=perdida, logits=logits.detach(), objetivos=y,
                    datos_extra={"prediccion": logits.detach().argmax(1), "real": y})

    def evaluador(self, info: InfoDatos):
        return EvaluadorSegmentacion(info.clases, self.cfg)

    def exportar_extra(self) -> dict:
        return {"arquitectura": self.cfg.modelo.arquitectura,
                "tam_img": self.cfg.datos.tam_img,
                "clases_segmentacion": self.clases}


def _cambiar_cabezas(modelo: nn.Module, num_clases: int) -> None:
    """Sustituye la última convolución de cada cabeza por una del nº de clases correcto."""
    for atributo in ("classifier", "aux_classifier"):
        cabeza = getattr(modelo, atributo, None)
        if cabeza is None:
            continue
        if hasattr(cabeza, "low_classifier"):          # LRASPP
            cabeza.low_classifier = nn.Conv2d(cabeza.low_classifier.in_channels,
                                              num_clases, 1)
            cabeza.high_classifier = nn.Conv2d(cabeza.high_classifier.in_channels,
                                               num_clases, 1)
            continue
        ultima = cabeza[-1]
        if isinstance(ultima, nn.Conv2d):
            cabeza[-1] = nn.Conv2d(ultima.in_channels, num_clases, ultima.kernel_size,
                                   stride=ultima.stride)


class EvaluadorSegmentacion:
    def __init__(self, clases: list[str], cfg):
        self.clases = clases
        self.cfg = cfg
        self.metrica_objetivo = cfg.entrenamiento.metrica_objetivo
        self.reiniciar()

    def reiniciar(self) -> None:
        n = len(self.clases)
        self.matriz = np.zeros((n, n), dtype=np.int64)
        self._perdida, self._n = 0.0, 0

    def actualizar(self, logits=None, objetivos=None, perdida=None, subgrupos=None,
                   datos_extra=None) -> None:
        if perdida is not None:
            self._perdida += perdida
            self._n += 1
        if not datos_extra:
            return
        prediccion = datos_extra["prediccion"].cpu().numpy().ravel()
        real = datos_extra["real"].cpu().numpy().ravel()
        validos = real != datos_mod.IGNORAR
        prediccion, real = prediccion[validos], real[validos]
        n = len(self.clases)
        # Histograma 2D de (real, predicho): la matriz de confusión de todos los píxeles
        self.matriz += np.bincount(real * n + prediccion, minlength=n * n).reshape(n, n)

    def resumen(self, calibrar: bool = False, curvas: bool = True) -> dict:
        if not self.matriz.sum():
            return {}
        diagonal = np.diag(self.matriz).astype(float)
        union = self.matriz.sum(1) + self.matriz.sum(0) - diagonal
        iou = np.divide(diagonal, union, out=np.full_like(diagonal, np.nan),
                        where=union > 0)
        miou = float(np.nanmean(iou))
        acc_pixel = float(diagonal.sum() / self.matriz.sum())

        por_clase = {c: (round(float(iou[i]), 4) if not np.isnan(iou[i]) else None)
                     for i, c in enumerate(self.clases)}
        detalle = " · ".join(f"{c} {v:.3f}" for c, v in por_clase.items() if v is not None)
        return {
            "acc": miou, "acc_balanceada": miou, "miou": miou,
            "acc_pixel": acc_pixel, "n": int(self.matriz.sum()),
            "perdida": self._perdida / max(1, self._n),
            "iou_por_clase": por_clase,
            "texto": (f"mIoU {miou:.4f} · acierto por píxel {acc_pixel:.4f}\n"
                      f"IoU por clase: {detalle}"),
        }

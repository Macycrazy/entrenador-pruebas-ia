"""Detectores de torchvision con la cabeza cambiada al número de clases del dataset."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import detection

# nombre -> (constructor, familia de cabeza)
DETECTORES = {
    "fasterrcnn_resnet50_fpn_v2": "fasterrcnn",
    "fasterrcnn_resnet50_fpn": "fasterrcnn",
    "fasterrcnn_mobilenet_v3_large_fpn": "fasterrcnn",
    "fasterrcnn_mobilenet_v3_large_320_fpn": "fasterrcnn",   # el más rápido
    "retinanet_resnet50_fpn_v2": "retinanet",
    "retinanet_resnet50_fpn": "retinanet",
    "fcos_resnet50_fpn": "fcos",
}


def crear_modelo(cfg, num_clases: int) -> nn.Module:
    """num_clases incluye el fondo: 3 objetos distintos -> num_clases = 4."""
    nombre = cfg.modelo.arquitectura
    if nombre not in DETECTORES:
        raise SystemExit(f"Detector '{nombre}' desconocido. Opciones: "
                         f"{', '.join(sorted(DETECTORES))}")

    pesos = "DEFAULT" if cfg.modelo.preentrenado else None
    modelo = getattr(detection, nombre)(weights=pesos)
    familia = DETECTORES[nombre]

    if familia == "fasterrcnn":
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        entradas = modelo.roi_heads.box_predictor.cls_score.in_features
        modelo.roi_heads.box_predictor = FastRCNNPredictor(entradas, num_clases)

    elif familia == "retinanet":
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead
        cabeza = modelo.head.classification_head
        modelo.head.classification_head = RetinaNetClassificationHead(
            in_channels=modelo.backbone.out_channels,
            num_anchors=modelo.anchor_generator.num_anchors_per_location()[0],
            num_classes=num_clases,
            norm_layer=getattr(cabeza, "norm_layer", None),
        )

    elif familia == "fcos":
        from torchvision.models.detection.fcos import FCOSClassificationHead
        modelo.head.classification_head = FCOSClassificationHead(
            in_channels=modelo.backbone.out_channels,
            num_anchors=modelo.anchor_generator.num_anchors_per_location()[0],
            num_classes=num_clases,
        )

    return modelo


def listar() -> list[str]:
    return sorted(DETECTORES)

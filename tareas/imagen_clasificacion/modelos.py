"""Construcción del modelo: backbone (torchvision / timm / rostros) + cabeza(s)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from nucleo.perdidas import CabezaMargen


class Clasificador(nn.Module):
    """Backbone sin su clasificador original + cabeza propia (+ cabezas auxiliares).

    Separarlo así permite cambiar de cabeza (lineal o con margen) y añadir tareas
    auxiliares (edad, etnia) sin tocar el backbone.
    """

    def __init__(self, backbone: nn.Module, caracteristicas: int, clases: int, cfg,
                 extras: dict[str, int] | None = None):
        super().__init__()
        self.backbone = backbone
        self.caracteristicas = caracteristicas
        self.dropout = nn.Dropout(cfg.modelo.dropout)
        tipo = cfg.modelo.cabeza.lower()
        if tipo in ("arcface", "cosface"):
            self.cabeza = CabezaMargen(caracteristicas, clases, cfg.perdida.arcface_margen,
                                       cfg.perdida.arcface_escala, tipo)
            self.con_margen = True
        else:
            self.cabeza = nn.Linear(caracteristicas, clases)
            self.con_margen = False
        self.cabezas_extra = nn.ModuleDict({
            nombre: nn.Linear(caracteristicas, n) for nombre, n in (extras or {}).items()
        })

    def forward(self, x, objetivo=None, solo_logits: bool = True):
        rasgos = self.dropout(self.backbone(x))
        logits = self.cabeza(rasgos, objetivo) if self.con_margen else self.cabeza(rasgos)
        if solo_logits and not self.cabezas_extra:
            return logits
        extra = {nombre: cabeza(rasgos) for nombre, cabeza in self.cabezas_extra.items()}
        return {"principal": logits, "rasgos": rasgos, **extra}


def crear_modelo(cfg, clases: int, extras: dict[str, int] | None = None) -> Clasificador:
    nombre = cfg.modelo.arquitectura
    if nombre.startswith("timm:"):
        backbone, caracteristicas = _timm(nombre[5:], cfg)
    elif nombre.startswith("rostro:"):
        backbone, caracteristicas = _rostro(nombre[7:], cfg)
    else:
        backbone, caracteristicas = _torchvision(nombre, cfg)

    if cfg.modelo.checkpoint_gradiente:
        if hasattr(backbone, "set_grad_checkpointing"):
            backbone.set_grad_checkpointing(True)
        else:
            print("AVISO: checkpoint de gradiente solo está disponible en backbones timm.")
    return Clasificador(backbone, caracteristicas, clases, cfg, extras)


# ---------------------------------------------------------------- backbones

def _torchvision(nombre: str, cfg):
    if not hasattr(models, nombre):
        raise SystemExit(f"Arquitectura '{nombre}' no existe en torchvision. "
                         f"Prueba `--listar-modelos`.")
    pesos = models.get_model_weights(nombre).DEFAULT if cfg.modelo.preentrenado else None
    modelo = getattr(models, nombre)(weights=pesos)
    return modelo, _quitar_cabeza(modelo, nombre)


def _quitar_cabeza(modelo: nn.Module, nombre: str) -> int:
    """Sustituye el clasificador original por Identity y devuelve su nº de entradas."""
    for atributo in ("fc", "classifier", "head", "heads"):
        if not hasattr(modelo, atributo):
            continue
        modulo = getattr(modelo, atributo)
        if isinstance(modulo, nn.Linear):
            setattr(modelo, atributo, nn.Identity())
            return modulo.in_features
        if isinstance(modulo, nn.Sequential):
            for i in range(len(modulo) - 1, -1, -1):
                if isinstance(modulo[i], nn.Linear):
                    entradas = modulo[i].in_features
                    modulo[i] = nn.Identity()
                    return entradas
        if hasattr(modulo, "head") and isinstance(modulo.head, nn.Linear):   # ViT
            entradas = modulo.head.in_features
            modulo.head = nn.Identity()
            return entradas
    raise SystemExit(f"No sé dónde está el clasificador de '{nombre}'. "
                     "Usa una arquitectura de la lista o un modelo de timm.")


def _timm(nombre: str, cfg):
    try:
        import timm
    except ImportError:
        raise SystemExit("Los backbones timm necesitan:  pip install timm") from None
    modelo = timm.create_model(nombre, pretrained=cfg.modelo.preentrenado,
                               num_classes=0, drop_path_rate=cfg.modelo.drop_path)
    return modelo, modelo.num_features


def _rostro(nombre: str, cfg):
    """Backbones preentrenados con caras: mucho mejor punto de partida con pocos datos."""
    if nombre in ("facenet", "vggface2", "casia-webface"):
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError:
            raise SystemExit("Necesita:  pip install facenet-pytorch") from None
        pesos = "vggface2" if nombre in ("facenet", "vggface2") else "casia-webface"
        modelo = InceptionResnetV1(pretrained=pesos if cfg.modelo.preentrenado else None)
        modelo.logits = nn.Identity()
        return modelo, 512

    if nombre.startswith("insightface"):
        try:
            import insightface  # noqa: F401
        except ImportError:
            raise SystemExit("Necesita:  pip install insightface onnxruntime") from None
        raise SystemExit("Los modelos de insightface son solo de inferencia (ONNX). "
                         "Para fine-tune usa 'rostro:facenet' o 'timm:...'.")

    raise SystemExit(f"Backbone de rostros '{nombre}' desconocido "
                     "(opciones: facenet, casia-webface)")


def listar_arquitecturas() -> dict[str, list[str]]:
    torchvision_nombres = sorted(
        n for n in dir(models)
        if not n.startswith("_") and callable(getattr(models, n))
        and n[0].islower() and n not in ("get_model", "get_model_builder",
                                         "get_model_weights", "get_weight", "list_models")
    )
    salida = {"torchvision": torchvision_nombres, "rostro": ["facenet", "casia-webface"]}
    try:
        import timm
        salida["timm (prefijo timm:)"] = timm.list_models(pretrained=True)
    except ImportError:
        salida["timm (prefijo timm:)"] = ["(instala timm para ver ~1000 modelos)"]
    return salida


@torch.no_grad()
def probar_forma(modelo: nn.Module, tam: int = 224) -> tuple:
    modelo.eval()
    return tuple(modelo(torch.zeros(2, 3, tam, tam)).shape)

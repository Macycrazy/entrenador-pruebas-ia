"""Reconstruye un modelo entrenado a partir de su checkpoint, sea de la tarea que sea.

El checkpoint lleva dentro la configuración completa, así que no hace falta recordar
con qué arquitectura ni a qué resolución se entrenó.
"""

from __future__ import annotations

from pathlib import Path

import torch

from nucleo.config import Config


def cargar_modelo(ruta: str | Path, dispositivo: str = "cpu", con_ema: bool = True):
    ruta = Path(ruta)
    if ruta.is_dir():
        ruta = ruta / "mejor.pt"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el checkpoint {ruta}")

    ckpt = torch.load(ruta, map_location=dispositivo, weights_only=False)
    cfg = Config(ckpt.get("config", {}))
    tarea = ckpt.get("tarea", cfg.get("tarea", "imagen_clasificacion"))

    modelo = _constructor(tarea)(cfg, ckpt["clases"], ckpt)
    estado = ckpt.get("ema_state_dict") if con_ema and "ema_state_dict" in ckpt else ckpt["state_dict"]
    modelo.load_state_dict(estado)
    modelo.eval().to(dispositivo)
    return modelo, ckpt


def _constructor(tarea: str):
    if tarea in ("imagen_clasificacion", "rostro_antispoofing"):
        from tareas.imagen_clasificacion.modelos import crear_modelo

        def construir(cfg, clases, ckpt):
            extras = {k: len(v) for k, v in (ckpt.get("extras") or {}).items()}
            return crear_modelo(cfg, len(clases), extras)
        return construir

    if tarea == "rostro_identificacion":
        from tareas.rostro_identificacion.modelos import crear_modelo as crear

        return lambda cfg, clases, ckpt: crear(cfg, len(clases))

    raise SystemExit(f"No sé reconstruir modelos de la tarea '{tarea}'")


def metadatos(ckpt: dict) -> dict:
    """Lo que necesita un servidor de inferencia para preprocesar igual que el entrenamiento."""
    cfg = Config(ckpt.get("config", {}))
    return {
        "clases": ckpt["clases"],
        "tam_img": ckpt.get("tam_img", cfg.get("datos", {}).get("tam_img", 224)),
        "media": ckpt.get("media", (0.485, 0.456, 0.406)),
        "desv": ckpt.get("desv", (0.229, 0.224, 0.225)),
        "arquitectura": ckpt.get("arquitectura", cfg.get("modelo", {}).get("arquitectura", "?")),
        "tarea": ckpt.get("tarea", "imagen_clasificacion"),
        "fecha": ckpt.get("fecha", "?"),
        "metricas": ckpt.get("metricas", {}),
        "temperatura": ckpt.get("metricas", {}).get("temperatura", 1.0),
        "extras": ckpt.get("extras", {}),
    }

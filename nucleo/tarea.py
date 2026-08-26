"""Interfaz que debe cumplir cualquier tarea entrenable, y su registro.

El núcleo (bucle, optimizadores, métricas, checkpoints) no sabe nada de imágenes:
solo pide a la tarea sus datos, su modelo y cómo se ejecuta un lote. Añadir voz,
texto o detección es escribir una clase con estos cinco métodos.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn

REGISTRO: dict[str, type["Tarea"]] = {}

TAREAS_CONOCIDAS = {
    "imagen_clasificacion": "tareas.imagen_clasificacion",
    "rostro_identificacion": "tareas.rostro_identificacion",
    "rostro_antispoofing": "tareas.rostro_antispoofing",
    "vision_deteccion": "tareas.vision_deteccion",
    "vision_segmentacion": "tareas.vision_segmentacion",
    "imagen_superresolucion": "tareas.imagen_superresolucion",
    "imagen_generacion": "tareas.imagen_generacion",
    "imagen_anomalias": "tareas.imagen_anomalias",
    "audio_clasificacion": "tareas.audio_clasificacion",
    "texto_clasificacion": "tareas.texto_clasificacion",
    "texto_llm": "tareas.texto_llm",
    "texto_ner": "tareas.texto_ner",
    "tabular": "tareas.tabular",
    "series": "tareas.series",
    "voz_sintesis": "tareas.voz_sintesis",
    "audio_transcripcion": "tareas.audio_transcripcion",
}


def registrar(nombre: str):
    def decorador(clase):
        clase.nombre = nombre
        REGISTRO[nombre] = clase
        return clase
    return decorador


def crear_tarea(cfg) -> "Tarea":
    nombre = cfg.tarea
    if nombre not in REGISTRO:
        modulo = TAREAS_CONOCIDAS.get(nombre)
        if not modulo:
            raise SystemExit(f"Tarea '{nombre}' desconocida. "
                             f"Disponibles: {', '.join(sorted(TAREAS_CONOCIDAS))}")
        try:
            importlib.import_module(modulo)
        except ImportError as error:
            raise SystemExit(f"No se pudo cargar la tarea '{nombre}': {error}") from error
    return REGISTRO[nombre](cfg)


@dataclass
class Paso:
    """Resultado de ejecutar un lote."""
    perdida: torch.Tensor
    logits: torch.Tensor
    objetivos: torch.Tensor
    subgrupos: dict = field(default_factory=dict)
    # Cualquier cosa que el evaluador de la tarea necesite además de los logits
    # (por ejemplo los embeddings, para las métricas de verificación facial).
    datos_extra: dict = field(default_factory=dict)


@dataclass
class InfoDatos:
    clases: list[str]
    conteo: list[int]
    n_train: int
    n_val: int
    subgrupos: list[str] = field(default_factory=list)

    def resumen(self) -> str:
        # Con muchas clases (identidades) el detalle completo es un muro de texto.
        if len(self.clases) > 12:
            ejemplos = ", ".join(self.clases[:3])
            return (f"train {self.n_train} · val {self.n_val} · {len(self.clases)} clases "
                    f"({ejemplos}…) · fotos por clase: mín {min(self.conteo)}, "
                    f"máx {max(self.conteo)}, media {sum(self.conteo) / len(self.conteo):.1f}")
        detalle = ", ".join(f"{c}={n}" for c, n in zip(self.clases, self.conteo))
        return f"train {self.n_train} · val {self.n_val} · clases: {detalle}"


class Tarea(ABC):
    nombre: str = "sin_nombre"

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def datos(self) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, InfoDatos]:
        """Devuelve (loader de entrenamiento, loader de validación, info)."""

    @abstractmethod
    def modelo(self, info: InfoDatos) -> nn.Module:
        """Construye el modelo para esta tarea."""

    @abstractmethod
    def criterio(self, info: InfoDatos, dispositivo: str) -> nn.Module:
        """Función de pérdida ya configurada (pesos de clase incluidos si toca)."""

    @abstractmethod
    def paso(self, modelo: nn.Module, lote, criterio, dispositivo, entrenando: bool) -> Paso:
        """Ejecuta un lote: mueve datos, hace forward y calcula la pérdida."""

    @abstractmethod
    def evaluador(self, info: InfoDatos):
        """Objeto que acumula predicciones y produce el informe de métricas."""

    # --- opcionales -------------------------------------------------------

    def al_cambiar_epoca(self, epoca: int, modelo: nn.Module) -> None:
        """Gancho por época (resolución progresiva, descongelado gradual…)."""

    def descripcion(self) -> str:
        """Cómo se llama el modelo de esta tarea en los mensajes de la consola."""
        return self.cfg.modelo.arquitectura

    def exportar_extra(self) -> dict:
        """Metadatos que se guardan en el checkpoint (normalización, clases…)."""
        return {}

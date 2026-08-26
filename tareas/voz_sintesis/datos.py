"""Pares (texto, audio) para ajustar una voz.

    datos_voz/
        textos.csv        con columnas: archivo,texto
        audios/001.wav …

Grabaciones limpias de la misma persona: con 10-30 minutos el modelo aprende su timbre.
Cuanto más se parezca el estilo de las frases al uso final, mejor.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from tareas.audio_clasificacion.carga import cargar_audio


def recopilar(cfg) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    raiz = Path(cfg.datos.ruta)
    indice = raiz / "textos.csv"
    if not indice.exists():
        raise SystemExit(
            f"Falta {indice} con columnas 'archivo,texto'.\n"
            "Cada fila apunta a una grabación y dice qué se dice en ella.")

    with indice.open(encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    carpeta = raiz / "audios" if (raiz / "audios").is_dir() else raiz

    muestras = []
    for fila in filas:
        ruta = carpeta / fila["archivo"]
        if ruta.exists() and fila.get("texto", "").strip():
            muestras.append((ruta, fila["texto"].strip()))
    if not muestras:
        raise SystemExit(f"Ninguna fila de {indice} apunta a un audio existente")

    aleatorio = random.Random(cfg.semilla)
    aleatorio.shuffle(muestras)
    if cfg.datos.limite:
        muestras = muestras[:cfg.datos.limite]
    corte = max(1, int(len(muestras) * cfg.datos.val_proporcion))
    return muestras[corte:], muestras[:corte]


class DatasetVoz(Dataset):
    def __init__(self, muestras, procesador, vector_hablante, cfg):
        self.muestras = muestras
        self.procesador = procesador
        self.vector = vector_hablante
        self.cfg = cfg

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, indice):
        ruta, texto = self.muestras[indice]
        onda, sr = cargar_audio(ruta)
        if sr != self.cfg.voz.sr:
            import torchaudio
            onda = torchaudio.functional.resample(onda, sr, self.cfg.voz.sr)

        procesado = self.procesador(
            text=texto, audio_target=onda.squeeze(0).numpy(),
            sampling_rate=self.cfg.voz.sr, return_attention_mask=False)
        return {"input_ids": procesado["input_ids"],
                "labels": procesado["labels"][0],
                "speaker_embeddings": self.vector}


def crear_juntador(procesador):
    def juntar(lote):
        entradas = procesador.pad(
            input_ids=[{"input_ids": m["input_ids"]} for m in lote],
            labels=[{"input_values": m["labels"]} for m in lote],
            return_tensors="pt")
        # El modelo reduce el espectrograma por 2: la longitud debe ser múltiplo
        objetivos = entradas["labels"]
        if objetivos.shape[1] % 2:
            entradas["labels"] = objetivos[:, :-1]
            entradas["stop_labels"] = entradas["stop_labels"][:, :-1] \
                if "stop_labels" in entradas else None
        entradas["speaker_embeddings"] = torch.stack([m["speaker_embeddings"] for m in lote])
        entradas.pop("decoder_attention_mask", None)
        return entradas, entradas["labels"], {}, {}
    return juntar


def crear_loaders(cfg, train, val, procesador, vector):
    juntar = crear_juntador(procesador)
    comunes = dict(num_workers=cfg.datos.workers, collate_fn=juntar,
                   persistent_workers=cfg.datos.workers > 0)
    return (DataLoader(DatasetVoz(train, procesador, vector, cfg),
                       batch_size=cfg.datos.batch, shuffle=True, **comunes),
            DataLoader(DatasetVoz(val, procesador, vector, cfg),
                       batch_size=cfg.datos.batch, shuffle=False, **comunes))

"""Pares (audio, texto) para afinar Whisper. Mismo formato que la síntesis de voz:

    datos_voz/textos.csv   con columnas: archivo,texto
    datos_voz/audios/*.wav
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from tareas.audio_clasificacion.carga import cargar_audio
from tareas.voz_sintesis.datos import recopilar  # noqa: F401  (mismo formato)


class DatasetTranscripcion(Dataset):
    def __init__(self, muestras, procesador, cfg):
        self.muestras = muestras
        self.procesador = procesador
        self.cfg = cfg

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, indice):
        ruta, texto = self.muestras[indice]
        onda, sr = cargar_audio(ruta)
        objetivo = self.cfg.transcripcion.sr
        if sr != objetivo:
            import torchaudio
            onda = torchaudio.functional.resample(onda, sr, objetivo)
        onda = onda.squeeze(0)[:int(objetivo * self.cfg.transcripcion.duracion_max)]

        rasgos = self.procesador.feature_extractor(
            onda.numpy(), sampling_rate=objetivo, return_tensors="pt").input_features[0]
        etiquetas = self.procesador.tokenizer(texto).input_ids
        return {"input_features": rasgos, "labels": etiquetas}


def crear_juntador(procesador):
    def juntar(lote):
        rasgos = torch.stack([m["input_features"] for m in lote])
        etiquetas = procesador.tokenizer.pad(
            [{"input_ids": m["labels"]} for m in lote], return_tensors="pt")
        # -100 marca el relleno para que no cuente en la pérdida
        ids = etiquetas["input_ids"].masked_fill(etiquetas.attention_mask.ne(1), -100)
        return {"input_features": rasgos}, ids, {}, {}
    return juntar


def crear_loaders(cfg, train, val, procesador):
    juntar = crear_juntador(procesador)
    comunes = dict(num_workers=cfg.datos.workers, collate_fn=juntar,
                   persistent_workers=cfg.datos.workers > 0)
    return (DataLoader(DatasetTranscripcion(train, procesador, cfg),
                       batch_size=cfg.datos.batch, shuffle=True, **comunes),
            DataLoader(DatasetTranscripcion(val, procesador, cfg),
                       batch_size=cfg.datos.batch, shuffle=False, **comunes))

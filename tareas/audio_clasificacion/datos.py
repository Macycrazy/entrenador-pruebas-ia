"""Audio → espectrograma mel → «imagen» de 3 canales.

Con eso, todo el catálogo de backbones y casi todas las aumentaciones de visión
sirven para sonido. Es lo que se hace en la práctica y funciona muy bien para
clasificar voz, hablantes, palabras clave o ruidos de máquina.

    datos_audio/
        hombre/  a.wav b.wav …          (o las clases que sean)
        mujer/   …
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from tareas.imagen_clasificacion.datos import Muestra, juntar

from .carga import cargar_audio

EXTENSIONES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def recopilar(cfg) -> tuple[list[Muestra], list[Muestra], list[str]]:
    raiz = Path(cfg.datos.ruta)
    if not raiz.exists():
        raise SystemExit(
            f"No existe {raiz}. Estructura esperada: {raiz}/<clase>/*.wav\n"
            "Para probar sin datos:  python preparacion/generar_audio_sintetico.py")

    dir_train = raiz / "train" if (raiz / "train").exists() else raiz
    dir_val = raiz / "val" if (raiz / "val").exists() else None

    train, clases = _listar(dir_train, cfg.datos.clases)
    if dir_val:
        val, _ = _listar(dir_val, clases)
    else:
        aleatorio = random.Random(cfg.semilla)
        aleatorio.shuffle(train)
        corte = max(1, int(len(train) * cfg.datos.val_proporcion))
        train, val = train[corte:], train[:corte]

    if cfg.datos.limite:
        train = train[:cfg.datos.limite]
        val = val[:max(1, cfg.datos.limite // 5)]
    return train, val, clases


def _listar(raiz: Path, clases):
    carpetas = sorted(p.name for p in raiz.iterdir() if p.is_dir())
    clases = list(clases) if clases else carpetas
    muestras = []
    for indice, clase in enumerate(clases):
        for ruta in sorted((raiz / clase).rglob("*")):
            if ruta.suffix.lower() in EXTENSIONES:
                muestras.append(Muestra(ruta, indice))
    if not muestras:
        raise SystemExit(f"No hay audios en {raiz} (extensiones: {', '.join(EXTENSIONES)})")
    return muestras, clases


class DatasetAudio(Dataset):
    def __init__(self, muestras: list[Muestra], cfg, entrenando: bool):
        import torchaudio

        self.muestras = muestras
        self.cfg = cfg
        self.entrenando = entrenando
        a = cfg.audio
        self.muestras_por_clip = int(a.sr * a.duracion)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=a.sr, n_fft=a.n_fft, hop_length=a.hop, n_mels=a.n_mels)
        self.a_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self._remuestrear = torchaudio.functional.resample

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, indice):
        muestra = self.muestras[indice]
        onda, sr = cargar_audio(muestra.ruta)                   # ya viene en mono
        if sr != self.cfg.audio.sr:
            onda = self._remuestrear(onda, sr, self.cfg.audio.sr)
        onda = self._ajustar_duracion(onda)

        espectro = self.a_db(self.mel(onda))                    # (1, n_mels, tiempo)
        espectro = (espectro - espectro.mean()) / (espectro.std() + 1e-5)
        if self.entrenando:
            espectro = self._specaugment(espectro)
        return espectro.repeat(3, 1, 1), muestra.etiqueta, {}, {}

    def _ajustar_duracion(self, onda: torch.Tensor) -> torch.Tensor:
        objetivo = self.muestras_por_clip
        largo = onda.shape[-1]
        if largo < objetivo:
            return torch.nn.functional.pad(onda, (0, objetivo - largo))
        if largo == objetivo:
            return onda
        inicio = random.randint(0, largo - objetivo) if \
            (self.entrenando and self.cfg.audio.recorte_aleatorio) else (largo - objetivo) // 2
        return onda[..., inicio:inicio + objetivo]

    def _specaugment(self, espectro: torch.Tensor) -> torch.Tensor:
        """Tapa una banda de frecuencia y un tramo de tiempo: es el 'random erasing' del audio."""
        a = self.cfg.audio
        _, n_mels, n_tiempo = espectro.shape
        if a.specaug_frec > 0:
            ancho = random.randint(0, max(1, int(n_mels * a.specaug_frec)))
            inicio = random.randint(0, max(0, n_mels - ancho))
            espectro[:, inicio:inicio + ancho, :] = 0
        if a.specaug_tiempo > 0:
            ancho = random.randint(0, max(1, int(n_tiempo * a.specaug_tiempo)))
            inicio = random.randint(0, max(0, n_tiempo - ancho))
            espectro[:, :, inicio:inicio + ancho] = 0
        return espectro


def crear_loaders(cfg, train, val):
    ds_train = DatasetAudio(train, cfg, True)
    ds_val = DatasetAudio(val, cfg, False)
    comunes = dict(num_workers=cfg.datos.workers, collate_fn=juntar,
                   pin_memory=torch.cuda.is_available(),
                   persistent_workers=cfg.datos.workers > 0)

    muestreador = None
    if cfg.datos.balanceo == "sampler":
        conteo = Counter(m.etiqueta for m in train)
        pesos_clase = {c: len(train) / (len(conteo) * n) for c, n in conteo.items()}
        muestreador = WeightedRandomSampler([pesos_clase[m.etiqueta] for m in train],
                                            num_samples=len(train), replacement=True)

    return (DataLoader(ds_train, batch_size=cfg.datos.batch, shuffle=muestreador is None,
                       sampler=muestreador, drop_last=len(train) > cfg.datos.batch, **comunes),
            DataLoader(ds_val, batch_size=cfg.datos.batch * 2, shuffle=False, **comunes),
            ds_train, ds_val)

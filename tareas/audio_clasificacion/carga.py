"""Lectura de audio con alternativas.

torchaudio 2.11 dejó de traer decodificadores propios y delega en torchcodec, que a su
vez necesita FFmpeg instalado en el sistema. Para no depender de eso se intentan varias
vías en orden y se usa la primera que funcione.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch

_AVISADO = set()


def cargar_audio(ruta: str | Path) -> tuple[torch.Tensor, int]:
    """Devuelve (onda mono en float32 con forma (1, n), frecuencia de muestreo)."""
    ruta = Path(ruta)

    onda = _con_soundfile(ruta)
    if onda is None:
        onda = _con_torchaudio(ruta)
    if onda is None and ruta.suffix.lower() == ".wav":
        onda = _con_wave(ruta)
    if onda is None:
        raise SystemExit(
            f"No se pudo leer {ruta.name}. Instala un decodificador:\n"
            "  pip install soundfile          (recomendado: wav, flac, ogg y mp3 recientes)\n"
            "  pip install torchcodec         (requiere FFmpeg en el sistema)")
    return onda


def _con_soundfile(ruta: Path):
    try:
        import soundfile as sf
    except ImportError:
        return None
    try:
        datos, sr = sf.read(str(ruta), dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001 - formato no soportado por libsndfile
        return None
    return torch.from_numpy(datos.T.copy()).mean(0, keepdim=True), sr


def _con_torchaudio(ruta: Path):
    try:
        import torchaudio
        onda, sr = torchaudio.load(str(ruta))
    except Exception as error:  # noqa: BLE001 - falta torchcodec/ffmpeg, o formato raro
        if "torchaudio" not in _AVISADO:
            _AVISADO.add("torchaudio")
            print(f"AVISO: torchaudio no puede decodificar ({type(error).__name__}); "
                  "se usa soundfile o el lector wav de la librería estándar.")
        return None
    return onda.mean(0, keepdim=True), sr


def _con_wave(ruta: Path):
    """Último recurso sin dependencias: WAV PCM de 8/16/32 bits."""
    try:
        with wave.open(str(ruta), "rb") as f:
            canales, ancho, sr, marcos = (f.getnchannels(), f.getsampwidth(),
                                          f.getframerate(), f.getnframes())
            crudo = f.readframes(marcos)
    except Exception:  # noqa: BLE001
        return None

    tipos = {1: np.uint8, 2: "<i2", 4: "<i4"}
    if ancho not in tipos:
        return None
    datos = np.frombuffer(crudo, dtype=tipos[ancho]).astype(np.float32)
    if ancho == 1:
        datos = (datos - 128) / 128.0
    else:
        datos /= float(2 ** (8 * ancho - 1))
    datos = datos.reshape(-1, canales).mean(axis=1)
    return torch.from_numpy(datos.copy()).unsqueeze(0), sr

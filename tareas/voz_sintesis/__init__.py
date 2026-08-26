"""Tarea: síntesis y clonación de voz.

Parte de **SpeechT5 (licencia MIT)**, que genera habla condicionada por un «vector de
hablante»: un resumen numérico del timbre de una persona. Eso permite tres cosas:

1. **Hablar con una voz preajustada** — sin entrenar nada, eligiendo del banco de voces.
2. **Clonar** — extrayendo el vector de una grabación de referencia de unos segundos.
3. **Ajustar la voz** — reentrenando con 10-30 minutos de grabaciones de esa persona,
   que es lo que de verdad la clava.

    pip install transformers datasets soundfile
    pip install speechbrain      # solo para clonar desde una grabación

Los modelos de mejor calidad (XTTS-v2, F5-TTS) son de uso NO comercial; por eso el
sistema usa SpeechT5, que es MIT y se puede reentrenar y desplegar sin ataduras.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from nucleo.tarea import InfoDatos, Paso, Tarea, registrar

from . import datos as datos_mod

RAIZ = Path(__file__).resolve().parent.parent.parent


@registrar("voz_sintesis")
class TareaVozSintesis(Tarea):

    def datos(self):
        from transformers import SpeechT5Processor

        self.procesador = SpeechT5Processor.from_pretrained(self.cfg.voz.modelo_base)
        self.vector = vector_hablante(self.cfg)
        train, val = datos_mod.recopilar(self.cfg)
        loader_train, loader_val = datos_mod.crear_loaders(
            self.cfg, train, val, self.procesador, self.vector)
        print(f"voz: {self.cfg.voz.modelo_base} · {len(train)} grabaciones de entrenamiento")
        return loader_train, loader_val, InfoDatos(
            clases=["voz"], conteo=[len(train)], n_train=len(train), n_val=len(val))

    def modelo(self, info: InfoDatos):
        from transformers import SpeechT5ForTextToSpeech
        modelo = SpeechT5ForTextToSpeech.from_pretrained(self.cfg.voz.modelo_base)
        modelo.config.use_cache = False
        return modelo

    def criterio(self, info: InfoDatos, dispositivo):
        return None       # la pérdida (espectrograma + parada) la calcula el propio modelo

    def paso(self, modelo, lote, criterio, dispositivo, entrenando: bool,
             espejo: bool = False) -> Paso:
        entradas, objetivos, _, _ = lote
        entradas = {k: v.to(dispositivo) for k, v in entradas.items()
                    if isinstance(v, torch.Tensor)}
        salida = modelo(**entradas)
        cuantas = torch.zeros(objetivos.size(0))
        return Paso(perdida=salida.loss, logits=cuantas, objetivos=cuantas)

    def evaluador(self, info: InfoDatos):
        return EvaluadorVoz()

    def descripcion(self) -> str:
        return f"{self.cfg.voz.modelo_base} (voz)"

    def exportar_extra(self) -> dict:
        return {"modelo_base": self.cfg.voz.modelo_base,
                "vocoder": self.cfg.voz.vocoder,
                "sr": self.cfg.voz.sr,
                "arquitectura": "speecht5_tts"}


class EvaluadorVoz:
    """En síntesis no hay 'acierto': se sigue la pérdida de reconstrucción.

    La calidad real solo la juzga el oído, así que la vista permite escuchar el
    resultado de cada modelo.
    """

    def __init__(self):
        self.metrica_objetivo = "acc"
        self.reiniciar()

    def reiniciar(self) -> None:
        self._suma, self._n = 0.0, 0

    def actualizar(self, logits=None, objetivos=None, perdida=None, subgrupos=None,
                   datos_extra=None) -> None:
        if perdida is not None:
            self._suma += perdida
            self._n += 1

    def resumen(self, calibrar: bool = False, curvas: bool = True) -> dict:
        if not self._n:
            return {}
        perdida = self._suma / self._n
        return {"acc": 1.0 / (1.0 + perdida), "acc_balanceada": 1.0 / (1.0 + perdida),
                "perdida": perdida, "n": self._n,
                "texto": f"pérdida de reconstrucción {perdida:.4f} "
                         f"(más baja = la voz se parece más)"}


# ---------------------------------------------------------------- vectores de hablante

def vector_hablante(cfg) -> torch.Tensor:
    """El vector de 512 números que define el timbre. De un archivo propio o del banco."""
    if cfg.voz.vector_hablante:
        ruta = Path(cfg.voz.vector_hablante)
        if not ruta.is_absolute():
            ruta = RAIZ / ruta
        return torch.tensor(np.load(ruta), dtype=torch.float32).reshape(512)
    return voz_del_banco(cfg.voz.banco_voces, cfg.voz.voz_por_defecto)


def voz_del_banco(banco: str, indice: int) -> torch.Tensor:
    """Un vector del banco de voces preajustadas.

    Se lee el parquet directamente en vez de usar `datasets`: ese dataset se publica con
    un script de carga que las versiones nuevas de la librería ya no admiten.
    """
    tabla = _descargar_banco(banco)
    indice = min(max(0, indice), tabla.num_rows - 1)
    vector = tabla.column("xvector")[indice].as_py()
    return torch.tensor(vector, dtype=torch.float32)


def voces_disponibles(banco: str) -> int:
    return _descargar_banco(banco).num_rows


_BANCO = {}


def _descargar_banco(banco: str):
    import json
    import urllib.request

    if banco in _BANCO:
        return _BANCO[banco]
    import pyarrow.parquet as pq

    destino = RAIZ / "modelos" / f"{banco.replace('/', '_')}.parquet"
    if not destino.exists():
        url_api = f"https://huggingface.co/api/datasets/{banco}/parquet"
        with urllib.request.urlopen(url_api, timeout=60) as respuesta:
            listado = json.load(respuesta)
        split = next(iter(next(iter(listado.values())).values()))
        destino.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando el banco de voces ({banco})…")
        urllib.request.urlretrieve(split[0], destino)
    _BANCO[banco] = pq.read_table(destino, columns=["filename", "xvector"])
    return _BANCO[banco]


def extraer_vector(ruta_audio, sr_objetivo: int = 16000) -> np.ndarray:
    """Saca el vector de timbre de una grabación de referencia: esto es «clonar»."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        try:
            from speechbrain.pretrained import EncoderClassifier
        except ImportError:
            raise SystemExit(
                "Clonar desde una grabación necesita:  pip install speechbrain") from None

    from tareas.audio_clasificacion.carga import cargar_audio

    codificador = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir=str(RAIZ / "modelos" / "spkrec-xvect"))
    onda, sr = cargar_audio(ruta_audio)
    if sr != sr_objetivo:
        import torchaudio
        onda = torchaudio.functional.resample(onda, sr, sr_objetivo)
    with torch.no_grad():
        vector = codificador.encode_batch(onda)
        vector = torch.nn.functional.normalize(vector, dim=2)
    return vector.squeeze().cpu().numpy()

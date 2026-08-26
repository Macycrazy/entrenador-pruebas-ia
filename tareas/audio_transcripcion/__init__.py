"""Tarea: transcripción de voz a texto, afinando Whisper.

Whisper ya transcribe bien de fábrica, así que aquí el ajuste sirve para lo que el modelo
base hace peor: acento venezolano, vocabulario propio (nombres de gerencias, siglas del
CIIP) y grabaciones con el ruido de tu oficina.

La métrica es **WER** (tasa de error por palabra): 0,15 significa que se equivoca en 15
de cada 100 palabras. Más bajo es mejor, al revés que el acierto.

    pip install transformers soundfile
"""

from __future__ import annotations

import torch

from nucleo.tarea import InfoDatos, Paso, Tarea, registrar

from . import datos as datos_mod


@registrar("audio_transcripcion")
class TareaTranscripcion(Tarea):

    def datos(self):
        from transformers import WhisperProcessor

        t = self.cfg.transcripcion
        self.procesador = WhisperProcessor.from_pretrained(
            t.modelo_base, language=t.idioma, task=t.tarea)
        train, val = datos_mod.recopilar(self.cfg)
        loader_train, loader_val = datos_mod.crear_loaders(
            self.cfg, train, val, self.procesador)
        print(f"transcripción: {t.modelo_base} · {t.idioma} · {len(train)} grabaciones")
        return loader_train, loader_val, InfoDatos(
            clases=["texto"], conteo=[len(train)], n_train=len(train), n_val=len(val))

    def modelo(self, info: InfoDatos):
        from transformers import WhisperForConditionalGeneration

        modelo = WhisperForConditionalGeneration.from_pretrained(
            self.cfg.transcripcion.modelo_base)
        modelo.config.forced_decoder_ids = None
        modelo.config.suppress_tokens = []
        modelo.config.use_cache = False
        return modelo

    def criterio(self, info: InfoDatos, dispositivo):
        return None      # Whisper calcula su propia pérdida con las etiquetas

    def paso(self, modelo, lote, criterio, dispositivo, entrenando: bool,
             espejo: bool = False) -> Paso:
        entradas, etiquetas, _, _ = lote
        rasgos = entradas["input_features"].to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)
        salida = modelo(input_features=rasgos, labels=etiquetas)

        extra = {}
        if not entrenando:
            # En validación se genera el texto para poder medir el WER de verdad
            base = getattr(modelo, "_orig_mod", modelo)
            predicho = base.generate(input_features=rasgos, max_new_tokens=180)
            extra = {"predicho": predicho.detach().cpu(), "referencia": etiquetas.cpu()}

        cuantas = torch.zeros(etiquetas.size(0))
        return Paso(perdida=salida.loss, logits=cuantas, objetivos=cuantas, datos_extra=extra)

    def evaluador(self, info: InfoDatos):
        return EvaluadorTranscripcion(self.procesador)

    def descripcion(self) -> str:
        return f"{self.cfg.transcripcion.modelo_base} ({self.cfg.transcripcion.idioma})"

    def exportar_extra(self) -> dict:
        return {"modelo_base": self.cfg.transcripcion.modelo_base,
                "idioma": self.cfg.transcripcion.idioma,
                "arquitectura": "whisper"}


class EvaluadorTranscripcion:
    def __init__(self, procesador):
        self.procesador = procesador
        self.metrica_objetivo = "acc"
        self.reiniciar()

    def reiniciar(self) -> None:
        self.errores, self.palabras, self._perdida, self._n = 0, 0, 0.0, 0
        self.ejemplos = []

    def actualizar(self, logits=None, objetivos=None, perdida=None, subgrupos=None,
                   datos_extra=None) -> None:
        if perdida is not None:
            self._perdida += perdida
            self._n += 1
        if not datos_extra:
            return

        etiquetas = datos_extra["referencia"].clone()
        etiquetas[etiquetas == -100] = self.procesador.tokenizer.pad_token_id
        referencias = self.procesador.batch_decode(etiquetas, skip_special_tokens=True)
        predichos = self.procesador.batch_decode(datos_extra["predicho"],
                                                 skip_special_tokens=True)
        for referencia, predicho in zip(referencias, predichos):
            errores, palabras = _distancia_palabras(referencia, predicho)
            self.errores += errores
            self.palabras += palabras
            if len(self.ejemplos) < 3:
                self.ejemplos.append((referencia.strip(), predicho.strip()))

    def resumen(self, calibrar: bool = False, curvas: bool = True) -> dict:
        if not self._n:
            return {}
        wer = self.errores / self.palabras if self.palabras else 1.0
        muestra = "\n".join(f'  real:     "{r[:70]}"\n  predicho: "{p[:70]}"'
                            for r, p in self.ejemplos)
        return {
            # 'acc' = 1 - WER para que el núcleo siga eligiendo "más alto es mejor"
            "acc": max(0.0, 1 - wer), "acc_balanceada": max(0.0, 1 - wer),
            "wer": wer, "perdida": self._perdida / self._n, "n": self.palabras,
            "texto": f"WER {wer:.4f} ({self.errores} errores en {self.palabras} palabras)"
                     + (f"\n{muestra}" if muestra else ""),
        }


def _distancia_palabras(referencia: str, hipotesis: str) -> tuple[int, int]:
    """Distancia de edición por palabras (Levenshtein), que es lo que mide el WER."""
    a = referencia.lower().split()
    b = hipotesis.lower().split()
    if not a:
        return len(b), max(1, len(b))

    previa = list(range(len(b) + 1))
    for i, palabra_a in enumerate(a, 1):
        actual = [i]
        for j, palabra_b in enumerate(b, 1):
            actual.append(min(previa[j] + 1,           # borrar
                              actual[j - 1] + 1,        # insertar
                              previa[j - 1] + (palabra_a != palabra_b)))   # sustituir
        previa = actual
    return previa[-1], len(a)

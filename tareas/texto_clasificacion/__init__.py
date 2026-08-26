"""Tarea: clasificación de texto (intención, sentimiento, categoría, spam…).

Usa un modelo de Hugging Face como base y le cambia la cabeza. Todo lo demás —
optimizadores, schedulers, EMA, early stopping, métricas, calibración — es el mismo
núcleo que las tareas de imagen.

    pip install transformers
"""

from __future__ import annotations

from collections import Counter

import torch

from nucleo.metricas import Evaluador
from nucleo.perdidas import crear_perdida, pesos_de_clase
from nucleo.tarea import InfoDatos, Paso, Tarea, registrar

from . import datos as datos_mod


@registrar("texto_clasificacion")
class TareaTextoClasificacion(Tarea):

    def datos(self):
        cfg = self.cfg
        self.tokenizador = _tokenizador(cfg.texto.modelo_base)
        train, val, clases = datos_mod.recopilar(cfg)
        self.clases = clases
        loader_train, loader_val, self.ds_train, self.ds_val = datos_mod.crear_loaders(
            cfg, train, val, self.tokenizador)

        conteo = Counter(etiqueta for _, etiqueta in train)
        largos = [len(t.split()) for t, _ in train[:2000]]
        print(f"texto: {cfg.texto.modelo_base} · máximo {cfg.texto.longitud_max} tokens · "
              f"media de {sum(largos) / max(1, len(largos)):.0f} palabras por muestra")
        return loader_train, loader_val, InfoDatos(
            clases=clases, conteo=[conteo.get(i, 0) for i in range(len(clases))],
            n_train=len(train), n_val=len(val))

    def modelo(self, info: InfoDatos):
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained(
            self.cfg.texto.modelo_base, num_labels=len(info.clases))

    def criterio(self, info: InfoDatos, dispositivo):
        pesos = None
        if self.cfg.datos.balanceo == "pesos_perdida" or self.cfg.perdida.pesos_clase:
            pesos = pesos_de_clase(info.conteo, dispositivo)
        return crear_perdida(self.cfg, pesos).to(dispositivo)

    def paso(self, modelo, lote, criterio, dispositivo, entrenando: bool,
             espejo: bool = False) -> Paso:
        entradas, y, _, _ = lote
        entradas = {k: v.to(dispositivo) for k, v in entradas.items()}
        y = y.to(dispositivo)
        logits = modelo(**entradas).logits
        return Paso(perdida=criterio(logits, y), logits=logits.detach(), objetivos=y)

    def evaluador(self, info: InfoDatos):
        return Evaluador(info.clases, self.cfg.entrenamiento.metrica_objetivo)

    def descripcion(self) -> str:
        return self.cfg.texto.modelo_base

    def exportar_extra(self) -> dict:
        return {"modelo_base": self.cfg.texto.modelo_base,
                "longitud_max": self.cfg.texto.longitud_max}


def _tokenizador(nombre: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit("La tarea de texto necesita:  pip install transformers") from None
    return AutoTokenizer.from_pretrained(nombre)

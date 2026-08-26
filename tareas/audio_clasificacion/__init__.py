"""Tarea: clasificación de audio (voz, hablante, palabra clave, sonidos).

Reaprovecha el modelo, la pérdida y las métricas de clasificación de imagen: lo único
propio es convertir el sonido en espectrograma. Con esto se entrena, por ejemplo,
«género por voz», «¿quién habla?» o «¿ha sonado la alarma?».
"""

from __future__ import annotations

from collections import Counter

import torch

from nucleo.metricas import Evaluador
from nucleo.perdidas import crear_perdida, mezclar_lote, perdida_mezclada, pesos_de_clase
from nucleo.tarea import InfoDatos, Paso, Tarea, registrar
from tareas.imagen_clasificacion.modelos import crear_modelo

from . import datos as datos_mod


@registrar("audio_clasificacion")
class TareaAudioClasificacion(Tarea):

    def datos(self):
        train, val, clases = datos_mod.recopilar(self.cfg)
        self.clases = clases
        loader_train, loader_val, self.ds_train, self.ds_val = datos_mod.crear_loaders(
            self.cfg, train, val)

        conteo = Counter(m.etiqueta for m in train)
        a = self.cfg.audio
        print(f"audio: {a.sr} Hz · {a.duracion}s por muestra · {a.n_mels} bandas mel")
        return loader_train, loader_val, InfoDatos(
            clases=clases, conteo=[conteo.get(i, 0) for i in range(len(clases))],
            n_train=len(train), n_val=len(val))

    def modelo(self, info: InfoDatos):
        return crear_modelo(self.cfg, len(info.clases))

    def criterio(self, info: InfoDatos, dispositivo):
        pesos = None
        if self.cfg.datos.balanceo == "pesos_perdida" or self.cfg.perdida.pesos_clase:
            pesos = pesos_de_clase(info.conteo, dispositivo)
        return crear_perdida(self.cfg, pesos).to(dispositivo)

    def paso(self, modelo, lote, criterio, dispositivo, entrenando: bool,
             espejo: bool = False) -> Paso:
        x, y, _, _ = lote
        x = x.to(dispositivo, non_blocking=True)
        y = y.to(dispositivo, non_blocking=True)

        y_a, y_b, lam = y, y, 1.0
        if entrenando:
            # mixup en espectrogramas funciona igual de bien que en imágenes
            x, y_a, y_b, lam = mezclar_lote(x, y, self.cfg)

        logits = modelo(x)
        if isinstance(logits, dict):
            logits = logits["principal"]
        return Paso(perdida=perdida_mezclada(criterio, logits, y_a, y_b, lam),
                    logits=logits.detach(), objetivos=y)

    def evaluador(self, info: InfoDatos):
        return Evaluador(info.clases, self.cfg.entrenamiento.metrica_objetivo)

    def exportar_extra(self) -> dict:
        return {"arquitectura": self.cfg.modelo.arquitectura,
                "audio": dict(self.cfg.audio)}

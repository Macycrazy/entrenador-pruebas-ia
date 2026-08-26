"""Tarea: clasificación de imagen (género, edad, etnia, anti-spoofing, lo que sea).

Es la implementación de referencia de la interfaz `nucleo.tarea.Tarea`.
"""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F

from nucleo.metricas import Evaluador
from nucleo.perdidas import crear_perdida, mezclar_lote, perdida_mezclada, pesos_de_clase
from nucleo.tarea import InfoDatos, Paso, Tarea, registrar

from . import aumentos, datos as datos_mod, modelos

PESO_AUXILIAR = 0.3          # cuánto pesan las tareas extra (edad, etnia) frente a la principal


@registrar("imagen_clasificacion")
class TareaImagenClasificacion(Tarea):

    def datos(self):
        cfg = self.cfg
        train, val, clases = datos_mod.recopilar(cfg)
        self.clases = clases
        self.extras = datos_mod.vocabularios(train + val, cfg.datos.objetivos_extra)

        trans_train = aumentos.entrenamiento(cfg)
        trans_val = aumentos.validacion(cfg)
        loader_train, loader_val, self.ds_train, self.ds_val = datos_mod.crear_loaders(
            cfg, train, val, trans_train, trans_val, self.extras)

        conteo = Counter(m.etiqueta for m in train)
        info = InfoDatos(
            clases=clases,
            conteo=[conteo.get(i, 0) for i in range(len(clases))],
            n_train=len(train), n_val=len(val),
            subgrupos=list(cfg.datos.subgrupos),
        )
        if self.extras:
            detalle = ", ".join(f"{k} ({len(v)} valores)" for k, v in self.extras.items())
            print(f"Multitarea activa: {detalle}")
        return loader_train, loader_val, info

    def modelo(self, info: InfoDatos):
        tamanos = {nombre: len(vocab) for nombre, vocab in self.extras.items()}
        return modelos.crear_modelo(self.cfg, len(info.clases), tamanos)

    def criterio(self, info: InfoDatos, dispositivo):
        pesos = None
        if self.cfg.datos.balanceo == "pesos_perdida" or self.cfg.perdida.pesos_clase:
            pesos = pesos_de_clase(info.conteo, dispositivo)
        return crear_perdida(self.cfg, pesos).to(dispositivo)

    def paso(self, modelo, lote, criterio, dispositivo, entrenando: bool,
             espejo: bool = False) -> Paso:
        x, y, meta, extras = lote
        formato = torch.channels_last if self.cfg.entrenamiento.canales_last \
            else torch.contiguous_format
        x = x.to(dispositivo, non_blocking=True, memory_format=formato)
        y = y.to(dispositivo, non_blocking=True)
        if espejo:
            x = torch.flip(x, dims=[3])

        y_a, y_b, lam = y, y, 1.0
        if entrenando:
            x, y_a, y_b, lam = mezclar_lote(x, y, self.cfg)

        salida = self._forward(modelo, x, y_a if entrenando else None)
        logits = salida["principal"] if isinstance(salida, dict) else salida
        perdida = perdida_mezclada(criterio, logits, y_a, y_b, lam)

        if isinstance(salida, dict):
            perdida = perdida + self._perdida_auxiliar(salida, extras, dispositivo)
        if entrenando:
            perdida = perdida + self._perdida_destilacion(logits, x, dispositivo)

        return Paso(perdida=perdida, logits=logits.detach(), objetivos=y, subgrupos=meta)

    def evaluador(self, info: InfoDatos):
        return Evaluador(info.clases, self.cfg.entrenamiento.metrica_objetivo)

    def al_cambiar_epoca(self, epoca: int, modelo) -> None:
        """Resolución progresiva: entrenar pequeño al principio ahorra mucho tiempo."""
        escalones = self.cfg.entrenamiento.resolucion_progresiva
        if not escalones:
            return
        indice = min(len(escalones) - 1,
                     int(epoca / max(1, self.cfg.entrenamiento.epocas) * len(escalones)))
        tam = escalones[indice]
        if getattr(self, "_tam_actual", None) == tam:
            return
        self._tam_actual = tam
        self.ds_train.cambiar_transformacion(aumentos.entrenamiento(self.cfg, tam))
        self.ds_val.cambiar_transformacion(aumentos.validacion(self.cfg, tam))
        print(f"  resolución → {tam}px")

    def exportar_extra(self) -> dict:
        return {
            "arquitectura": self.cfg.modelo.arquitectura,
            "tam_img": getattr(self, "_tam_actual", self.cfg.datos.tam_img),
            "media": aumentos.MEDIA,
            "desv": aumentos.DESV,
            "extras": {k: list(v) for k, v in getattr(self, "extras", {}).items()},
        }

    # ------------------------------------------------------------------ internos

    def _forward(self, modelo, x, objetivo):
        base = getattr(modelo, "_orig_mod", modelo)
        con_margen = getattr(base, "con_margen", False)
        tiene_extras = bool(getattr(base, "cabezas_extra", {}))
        if con_margen or tiene_extras:
            return modelo(x, objetivo, solo_logits=not tiene_extras)
        return modelo(x)

    def _perdida_auxiliar(self, salida: dict, extras: dict, dispositivo) -> torch.Tensor:
        total = torch.zeros((), device=dispositivo)
        for nombre, objetivo in extras.items():
            if nombre in salida:
                total = total + F.cross_entropy(
                    salida[nombre], objetivo.to(dispositivo), ignore_index=-100)
        return PESO_AUXILIAR * total

    def _perdida_destilacion(self, logits, x, dispositivo) -> torch.Tensor:
        d = self.cfg.perdida.destilacion
        if not d.activa:
            return torch.zeros((), device=dispositivo)
        profesor = self._cargar_profesor(dispositivo)
        with torch.no_grad():
            logits_profesor = profesor(x)
            if isinstance(logits_profesor, dict):
                logits_profesor = logits_profesor["principal"]
        blanda = F.kl_div(
            F.log_softmax(logits / d.temperatura, dim=1),
            F.softmax(logits_profesor / d.temperatura, dim=1),
            reduction="batchmean") * d.temperatura ** 2
        return d.alfa * blanda

    def _cargar_profesor(self, dispositivo):
        if getattr(self, "_profesor", None) is not None:
            return self._profesor
        from nucleo.config import Config
        ruta = self.cfg.perdida.destilacion.profesor
        if not ruta:
            raise SystemExit("La destilación necesita perdida.destilacion.profesor=<checkpoint>")
        ckpt = torch.load(ruta, map_location=dispositivo, weights_only=False)
        cfg_profesor = Config(ckpt["config"])
        modelo = modelos.crear_modelo(cfg_profesor, len(ckpt["clases"]))
        modelo.load_state_dict(ckpt["state_dict"])
        self._profesor = modelo.to(dispositivo).eval()
        for p in self._profesor.parameters():
            p.requires_grad_(False)
        print(f"Profesor cargado desde {ruta}")
        return self._profesor

"""Tarea: detección de objetos (dónde está cada cosa, no solo qué hay).

Usa los detectores de torchvision, que en modo entrenamiento devuelven directamente
el diccionario de pérdidas y en evaluación las cajas predichas. La métrica es **mAP**,
calculada aquí sin depender de pycocotools.
"""

from __future__ import annotations

import numpy as np
import torch

from nucleo.tarea import InfoDatos, Paso, Tarea, registrar

from . import datos as datos_mod, modelos


@registrar("vision_deteccion")
class TareaVisionDeteccion(Tarea):

    def datos(self):
        from pathlib import Path
        raiz = Path(self.cfg.datos.ruta)
        clases = datos_mod.leer_clases(raiz)
        train = datos_mod.recopilar(raiz, "train", clases)
        val = datos_mod.recopilar(raiz, "val", clases)
        self.clases = clases

        if self.cfg.datos.limite:
            train, val = train[:self.cfg.datos.limite], val[:max(1, self.cfg.datos.limite // 5)]

        loader_train, loader_val, self.ds_train, self.ds_val = datos_mod.crear_loaders(
            self.cfg, train, val)

        objetos = sum(len(m.cajas) for m in train)
        conteo = [0] * len(clases)
        for m in train:
            for c in m.clases:
                if 1 <= c <= len(clases):
                    conteo[c - 1] += 1
        print(f"{objetos} objetos etiquetados en {len(train)} imágenes de entrenamiento")
        return loader_train, loader_val, InfoDatos(
            clases=clases, conteo=conteo, n_train=len(train), n_val=len(val))

    def modelo(self, info: InfoDatos):
        return modelos.crear_modelo(self.cfg, len(info.clases) + 1)   # +1 por el fondo

    def criterio(self, info: InfoDatos, dispositivo):
        return None      # las pérdidas las calcula el propio detector

    def paso(self, modelo, lote, criterio, dispositivo, entrenando: bool,
             espejo: bool = False) -> Paso:
        imagenes, objetivos = lote
        imagenes = [x.to(dispositivo) for x in imagenes]
        objetivos_gpu = [{"boxes": t["boxes"].to(dispositivo),
                          "labels": t["labels"].to(dispositivo)} for t in objetivos]
        cuantas = torch.zeros(len(imagenes))

        if entrenando:
            perdidas = modelo(imagenes, objetivos_gpu)
            total = sum(perdidas.values())
            return Paso(perdida=total, logits=cuantas, objetivos=cuantas)

        # En evaluación el detector devuelve cajas, no pérdidas.
        predicciones = modelo(imagenes)
        return Paso(
            perdida=torch.zeros((), device=dispositivo),
            logits=cuantas, objetivos=cuantas,
            datos_extra={"predicciones": [_a_numpy(p) for p in predicciones],
                         "objetivos": [_a_numpy(t) for t in objetivos]},
        )

    def evaluador(self, info: InfoDatos):
        return EvaluadorDeteccion(info.clases, self.cfg)

    def exportar_extra(self) -> dict:
        return {"arquitectura": self.cfg.modelo.arquitectura,
                "clases_deteccion": self.clases}


def _a_numpy(d: dict) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in d.items()
            if isinstance(v, torch.Tensor) and k in ("boxes", "labels", "scores")}


class EvaluadorDeteccion:
    """mAP al estilo COCO: media de la precisión media sobre umbrales de solapamiento."""

    def __init__(self, clases: list[str], cfg):
        self.clases = clases
        self.cfg = cfg
        self.metrica_objetivo = cfg.entrenamiento.metrica_objetivo
        self.reiniciar()

    def reiniciar(self) -> None:
        self.predicciones: list[dict] = []
        self.objetivos: list[dict] = []

    def actualizar(self, logits=None, objetivos=None, perdida=None, subgrupos=None,
                   datos_extra=None) -> None:
        if not datos_extra:
            return
        self.predicciones += datos_extra.get("predicciones", [])
        self.objetivos += datos_extra.get("objetivos", [])

    def resumen(self, calibrar: bool = False, curvas: bool = True) -> dict:
        if not self.objetivos:
            return {}
        umbrales = np.arange(0.5, 1.0, 0.05)
        aps = {u: calcular_ap(self.predicciones, self.objetivos, len(self.clases), u)
               for u in umbrales}

        map50 = float(np.nanmean(aps[umbrales[0]])) if len(aps[umbrales[0]]) else 0.0
        map5095 = float(np.nanmean([np.nanmean(v) for v in aps.values() if len(v)]))
        por_clase = {c: round(float(aps[umbrales[0]][i]), 4)
                     for i, c in enumerate(self.clases) if i < len(aps[umbrales[0]])}

        detalle = " · ".join(f"{c} {v:.3f}" for c, v in por_clase.items())
        return {
            "acc": map50, "acc_balanceada": map50, "map50": map50, "map": map5095,
            "n": len(self.objetivos), "perdida": 0.0,
            "ap_por_clase": por_clase,
            "texto": (f"mAP@0.5 {map50:.4f} · mAP@0.5:0.95 {map5095:.4f} · "
                      f"{len(self.objetivos)} imágenes\nAP por clase: {detalle}"),
        }


def calcular_ap(predicciones: list[dict], objetivos: list[dict], num_clases: int,
                umbral_iou: float) -> np.ndarray:
    """Precisión media por clase con interpolación de todos los puntos (estilo VOC 2010)."""
    aps = np.full(num_clases, np.nan)

    for indice in range(num_clases):
        clase = indice + 1                     # 0 es el fondo
        detecciones = []
        total_reales = 0
        reales_por_imagen = {}

        for i, objetivo in enumerate(objetivos):
            mascara = objetivo["labels"] == clase
            cajas = objetivo["boxes"][mascara]
            reales_por_imagen[i] = [cajas, np.zeros(len(cajas), dtype=bool)]
            total_reales += len(cajas)

        for i, prediccion in enumerate(predicciones):
            mascara = prediccion["labels"] == clase
            for caja, score in zip(prediccion["boxes"][mascara], prediccion["scores"][mascara]):
                detecciones.append((float(score), i, caja))

        if total_reales == 0:
            continue
        if not detecciones:
            aps[indice] = 0.0
            continue

        detecciones.sort(key=lambda d: -d[0])
        vp = np.zeros(len(detecciones))
        fp = np.zeros(len(detecciones))

        for j, (_, imagen, caja) in enumerate(detecciones):
            reales, usadas = reales_por_imagen[imagen]
            if len(reales) == 0:
                fp[j] = 1
                continue
            solapes = iou(caja, reales)
            mejor = int(np.argmax(solapes))
            if solapes[mejor] >= umbral_iou and not usadas[mejor]:
                vp[j] = 1
                usadas[mejor] = True
            else:
                fp[j] = 1

        vp_acum, fp_acum = np.cumsum(vp), np.cumsum(fp)
        recall = vp_acum / total_reales
        precision = vp_acum / np.maximum(vp_acum + fp_acum, 1e-9)
        # Precisión monótona decreciente, después área bajo la curva
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        aps[indice] = float(np.sum(np.diff(np.concatenate([[0], recall])) * precision))

    return aps


def iou(caja: np.ndarray, cajas: np.ndarray) -> np.ndarray:
    if len(cajas) == 0:
        return np.zeros(0)
    x1 = np.maximum(caja[0], cajas[:, 0])
    y1 = np.maximum(caja[1], cajas[:, 1])
    x2 = np.minimum(caja[2], cajas[:, 2])
    y2 = np.minimum(caja[3], cajas[:, 3])
    interseccion = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (caja[2] - caja[0]) * (caja[3] - caja[1])
    areas_b = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
    return interseccion / np.maximum(area_a + areas_b - interseccion, 1e-9)

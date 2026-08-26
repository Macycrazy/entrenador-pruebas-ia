"""Tarea: reconocimiento facial (identificación 1:N y verificación 1:1).

Entrena con una cabeza de margen sobre identidades y evalúa con las dos métricas
que de verdad importan en un control de acceso:

- **rank-1**: de una foto nueva de alguien conocido, ¿acierta quién es?
- **verificación**: dadas dos fotos, ¿son la misma persona? Se reporta EER y
  TAR@FAR (cuántos aciertos hay cuando solo se tolera 1 impostor de cada 1 000).
"""

from __future__ import annotations

import numpy as np
import torch

from nucleo.metricas import _auc
from nucleo.perdidas import crear_perdida
from nucleo.tarea import InfoDatos, Paso, Tarea, registrar
from tareas.imagen_clasificacion import aumentos
from tareas.imagen_clasificacion.datos import crear_loaders

from . import datos as datos_mod, modelos


@registrar("rostro_identificacion")
class TareaRostroIdentificacion(Tarea):

    def datos(self):
        cfg = self.cfg
        train, val, clases = datos_mod.recopilar(cfg)
        self.clases = clases
        loader_train, loader_val, self.ds_train, self.ds_val = crear_loaders(
            cfg, train, val, aumentos.entrenamiento(cfg), aumentos.validacion(cfg), {})

        conteo = [0] * len(clases)
        for m in train:
            conteo[m.etiqueta] += 1
        return loader_train, loader_val, InfoDatos(
            clases=clases, conteo=conteo, n_train=len(train), n_val=len(val))

    def modelo(self, info: InfoDatos):
        return modelos.crear_modelo(self.cfg, len(info.clases))

    def criterio(self, info: InfoDatos, dispositivo):
        return crear_perdida(self.cfg).to(dispositivo)

    def paso(self, modelo, lote, criterio, dispositivo, entrenando: bool,
             espejo: bool = False) -> Paso:
        x, y, _, _ = lote
        formato = torch.channels_last if self.cfg.entrenamiento.canales_last \
            else torch.contiguous_format
        x = x.to(dispositivo, non_blocking=True, memory_format=formato)
        y = y.to(dispositivo, non_blocking=True)
        if espejo:
            x = torch.flip(x, dims=[3])

        # El margen solo se aplica entrenando: en validación se mide el coseno limpio.
        salida = modelo(x, y if entrenando else None)
        logits, emb = salida["principal"], salida["rasgos"]
        return Paso(perdida=criterio(logits, y), logits=logits.detach(), objetivos=y,
                    datos_extra={"embeddings": emb.detach()})

    def evaluador(self, info: InfoDatos):
        return EvaluadorRostros(info.clases, self.cfg)

    def exportar_extra(self) -> dict:
        return {
            "arquitectura": self.cfg.modelo.arquitectura,
            "tam_img": self.cfg.datos.tam_img,
            "media": aumentos.MEDIA,
            "desv": aumentos.DESV,
            "dim_embedding": self.cfg.rostros.dim_embedding,
        }


class EvaluadorRostros:
    """Rank-1 desde los logits y verificación desde los embeddings."""

    def __init__(self, clases: list[str], cfg):
        self.clases, self.cfg = clases, cfg
        self.metrica_objetivo = cfg.entrenamiento.metrica_objetivo
        self.reiniciar()

    def reiniciar(self) -> None:
        self._aciertos = self._n = 0
        self._perdida = 0.0
        self._emb: list[np.ndarray] = []
        self._etiquetas: list[np.ndarray] = []

    def actualizar(self, logits, objetivos, perdida=None, subgrupos=None,
                   datos_extra=None) -> None:
        self._aciertos += int((logits.argmax(1) == objetivos).sum().item())
        self._n += objetivos.size(0)
        if perdida is not None:
            self._perdida += perdida * objetivos.size(0)
        emb = (datos_extra or {}).get("embeddings")
        if emb is not None:
            self._emb.append(emb.float().cpu().numpy())
            self._etiquetas.append(objetivos.cpu().numpy())

    def resumen(self, calibrar: bool = False, curvas: bool = True) -> dict:
        if not self._n:
            return {}
        rank1 = self._aciertos / self._n
        salida = {"acc": rank1, "acc_balanceada": rank1, "n": self._n,
                  "rank1": rank1, "perdida": self._perdida / self._n}

        if self._emb:
            embeddings = np.concatenate(self._emb)
            etiquetas = np.concatenate(self._etiquetas)
            salida.update(self._verificacion(embeddings, etiquetas))

        salida["texto"] = (
            f"rank-1 {rank1:.4f}"
            + (f" · verificación: AUC {salida['auc']:.4f} · EER {salida['eer']:.4f}"
               f" · TAR@FAR={self.cfg.rostros.far_objetivo:g} {salida['tar']:.4f}"
               f" (umbral coseno {salida['umbral']:.3f}) · {salida['pares']} pares"
               if "eer" in salida else " · sin pares suficientes para verificación"))
        return salida

    def _verificacion(self, embeddings: np.ndarray, etiquetas: np.ndarray) -> dict:
        similitudes, iguales = _construir_pares(
            embeddings, etiquetas, self.cfg.rostros.pares_max, self.cfg.semilla)
        if similitudes.size == 0 or iguales.sum() == 0 or (1 - iguales).sum() == 0:
            return {}

        orden = np.argsort(-similitudes)
        etiquetas_ordenadas = iguales[orden]
        positivos, negativos = iguales.sum(), (1 - iguales).sum()
        tar = np.cumsum(etiquetas_ordenadas) / positivos
        far = np.cumsum(1 - etiquetas_ordenadas) / negativos

        cruce = int(np.argmin(np.abs(far - (1 - tar))))
        eer = float((far[cruce] + (1 - tar[cruce])) / 2)

        objetivo = self.cfg.rostros.far_objetivo
        validos = np.where(far <= objetivo)[0]
        indice = int(validos[-1]) if validos.size else 0
        return {
            "auc": _auc(similitudes, iguales),
            "eer": eer,
            "tar": float(tar[indice]),
            "umbral": float(similitudes[orden][indice]),
            "umbral_eer": float(similitudes[orden][cruce]),
            "pares": int(similitudes.size),
        }


def _construir_pares(embeddings: np.ndarray, etiquetas: np.ndarray, maximo: int,
                     semilla: int) -> tuple[np.ndarray, np.ndarray]:
    """Pares positivos (misma persona) y otros tantos negativos, con coseno ya calculado."""
    aleatorio = np.random.default_rng(semilla)
    por_identidad: dict[int, list[int]] = {}
    for indice, etiqueta in enumerate(etiquetas):
        por_identidad.setdefault(int(etiqueta), []).append(indice)

    positivos = []
    for indices in por_identidad.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                positivos.append((indices[i], indices[j]))
    if not positivos:
        return np.array([]), np.array([])
    if len(positivos) > maximo // 2:
        elegidos = aleatorio.choice(len(positivos), maximo // 2, replace=False)
        positivos = [positivos[i] for i in elegidos]

    identidades = list(por_identidad)
    negativos = []
    intentos = 0
    while len(negativos) < len(positivos) and intentos < len(positivos) * 20:
        intentos += 1
        a, b = aleatorio.choice(len(identidades), 2, replace=False)
        ia, ib = identidades[int(a)], identidades[int(b)]
        negativos.append((int(aleatorio.choice(por_identidad[ia])),
                          int(aleatorio.choice(por_identidad[ib]))))

    pares = positivos + negativos
    iguales = np.array([1] * len(positivos) + [0] * len(negativos))
    a = embeddings[[p[0] for p in pares]]
    b = embeddings[[p[1] for p in pares]]
    similitudes = (a * b).sum(axis=1)     # los embeddings ya vienen normalizados
    return similitudes, iguales

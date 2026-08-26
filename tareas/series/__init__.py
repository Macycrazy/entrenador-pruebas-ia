"""Tarea: series temporales — predecir cómo sigue una serie de números.

Asistencia diaria, consumo eléctrico, trámites por semana, temperatura de una sala. Se le
dan los últimos N valores y devuelve los siguientes M.

    datos_series/serie.csv    con una columna de valores (y opcionalmente una de fecha)
    python entrenar.py --config configs/series.yaml --set series.columna=asistencia

La métrica es **MAPE**: el error medio en porcentaje, que se entiende sin saber la escala
de la serie. Se compara siempre contra la predicción ingenua (repetir el último valor),
porque muchas series se predicen sorprendentemente bien así y conviene saber si el modelo
aporta algo.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from nucleo.tarea import InfoDatos, Paso, Tarea, registrar
from tareas.tabular import _numero


@registrar("series")
class TareaSeries(Tarea):

    def datos(self):
        valores, fechas = _leer_serie(Path(self.cfg.datos.ruta), self.cfg.series.columna,
                                      self.cfg.series.fecha)
        s = self.cfg.series
        if len(valores) < s.ventana + s.horizonte + 10:
            raise SystemExit(
                f"La serie tiene {len(valores)} puntos; hacen falta al menos "
                f"{s.ventana + s.horizonte + 10} para ventana={s.ventana} y "
                f"horizonte={s.horizonte}")

        self.media = float(np.mean(valores))
        self.desv = float(np.std(valores)) or 1.0
        normal = (np.array(valores, dtype=np.float32) - self.media) / self.desv

        X, y = [], []
        for i in range(len(normal) - s.ventana - s.horizonte + 1):
            X.append(normal[i:i + s.ventana])
            y.append(normal[i + s.ventana:i + s.ventana + s.horizonte])
        X, y = np.stack(X), np.stack(y)

        # En series el corte es temporal, no aleatorio: validar con el futuro, no con
        # trozos intercalados, o el modelo estaría viendo lo que tiene que predecir.
        corte = int(len(X) * (1 - self.cfg.datos.val_proporcion))
        from torch.utils.data import DataLoader, TensorDataset
        def cargar(desde, hasta, mezclar):
            conjunto = TensorDataset(torch.tensor(X[desde:hasta]), torch.tensor(y[desde:hasta]))
            return DataLoader(conjunto, batch_size=self.cfg.datos.batch, shuffle=mezclar)

        print(f"serie «{s.columna}»: {len(valores)} puntos · ventana {s.ventana} → "
              f"horizonte {s.horizonte} · media {self.media:.2f}")
        self.fechas = fechas
        return cargar(0, corte, True), cargar(corte, len(X), False), InfoDatos(
            clases=["valor"], conteo=[corte], n_train=corte, n_val=len(X) - corte)

    def modelo(self, info: InfoDatos):
        s = self.cfg.series
        capas, dentro = [], s.ventana
        for ancho in s.capas:
            capas += [nn.Linear(dentro, ancho), nn.ReLU(), nn.Dropout(0.1)]
            dentro = ancho
        return nn.Sequential(*capas, nn.Linear(dentro, s.horizonte))

    def criterio(self, info: InfoDatos, dispositivo):
        return nn.L1Loss().to(dispositivo)

    def paso(self, modelo, lote, criterio, dispositivo, entrenando: bool,
             espejo: bool = False) -> Paso:
        x, y = lote
        x, y = x.to(dispositivo), y.to(dispositivo)
        prediccion = modelo(x)
        return Paso(perdida=criterio(prediccion, y), logits=prediccion.detach(), objetivos=y,
                    datos_extra={"ultimo": x[:, -1].detach()})

    def evaluador(self, info: InfoDatos):
        return EvaluadorSeries(self.media, self.desv)

    def descripcion(self) -> str:
        return f"serie {self.cfg.series.ventana}→{self.cfg.series.horizonte}"

    def exportar_extra(self) -> dict:
        return {"arquitectura": "mlp_series", "media": self.media, "desv": self.desv,
                "ventana": self.cfg.series.ventana, "horizonte": self.cfg.series.horizonte,
                "columna": self.cfg.series.columna}


class EvaluadorSeries:
    def __init__(self, media, desv):
        self.media, self.desv = media, desv
        self.metrica_objetivo = "acc"
        self.reiniciar()

    def reiniciar(self) -> None:
        self.error, self.error_ingenuo, self.suma, self.n = 0.0, 0.0, 0.0, 0
        self._perdida, self._lotes = 0.0, 0

    def actualizar(self, logits=None, objetivos=None, perdida=None, subgrupos=None,
                   datos_extra=None) -> None:
        if perdida is not None:
            self._perdida += perdida
            self._lotes += 1
        if logits is None:
            return
        # Se deshace la normalización para que el error esté en unidades reales
        p = logits.cpu().numpy() * self.desv + self.media
        r = objetivos.cpu().numpy() * self.desv + self.media
        self.error += float(np.abs(p - r).sum())
        self.suma += float(np.abs(r).sum())
        self.n += r.size
        if datos_extra is not None and "ultimo" in datos_extra:
            ingenua = datos_extra["ultimo"].cpu().numpy() * self.desv + self.media
            self.error_ingenuo += float(np.abs(ingenua[:, None] - r).sum())

    def resumen(self, calibrar: bool = False, curvas: bool = True) -> dict:
        if not self.n:
            return {}
        mae = self.error / self.n
        mape = self.error / max(1e-9, self.suma)
        mae_ingenuo = self.error_ingenuo / self.n
        return {"acc": max(0.0, 1 - mape), "acc_balanceada": max(0.0, 1 - mape),
                "mae": mae, "mape": mape, "mae_ingenuo": mae_ingenuo,
                "perdida": self._perdida / max(1, self._lotes), "n": self.n,
                "texto": f"error medio {mae:.3f} ({mape * 100:.1f} %) · "
                         f"repetir el último valor daría {mae_ingenuo:.3f}"}


def _leer_serie(ruta: Path, columna: str, columna_fecha: str):
    archivo = ruta / "serie.csv" if ruta.is_dir() else ruta
    if not archivo.exists():
        raise SystemExit(f"Falta {archivo} (un CSV con la serie)")
    with archivo.open(encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    if not filas:
        raise SystemExit(f"{archivo} está vacío")
    if not columna:
        raise SystemExit(
            "Indica qué columna predecir:  --set series.columna=<nombre>\n"
            f"Columnas: {', '.join(filas[0])}")
    if columna not in filas[0]:
        raise SystemExit(f"No existe la columna '{columna}'. Hay: {', '.join(filas[0])}")

    valores, fechas = [], []
    for fila in filas:
        numero = _numero(fila.get(columna))
        if numero is None:
            continue
        valores.append(numero)
        fechas.append(fila.get(columna_fecha, "") if columna_fecha else "")
    return valores, fechas

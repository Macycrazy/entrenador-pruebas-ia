"""Tarea: datos tabulares — predecir una columna a partir de las demás.

Es la IA menos vistosa y la más rentable: con el CSV que ya sale de cualquier sistema
—asistencia, nómina, trámites— se puede predecir ausentismo, rotación, retrasos o riesgo.

    datos_tabular/datos.csv    con cabecera; una fila por caso
    python entrenar.py --config configs/tabular.yaml --set tabular.objetivo=ausento

Detecta solo si la columna objetivo es una categoría (clasificación) o un número
(regresión), y usa la métrica adecuada: F1 balanceado o error medio absoluto.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from nucleo.tarea import InfoDatos, Paso, Tarea, registrar


@registrar("tabular")
class TareaTabular(Tarea):

    def datos(self):
        filas, columnas = _leer_csv(Path(self.cfg.datos.ruta))
        objetivo = self.cfg.tabular.objetivo
        if not objetivo:
            raise SystemExit(
                f"Falta indicar qué columna predecir:\n"
                f"  --set tabular.objetivo=<columna>\n"
                f"Columnas disponibles: {', '.join(columnas)}")
        if objetivo not in columnas:
            raise SystemExit(f"La columna '{objetivo}' no existe. Hay: {', '.join(columnas)}")

        entradas = [c for c in columnas
                    if c != objetivo and c not in (self.cfg.tabular.ignorar or [])]
        self.preparador = Preparador(filas, entradas, objetivo, self.cfg.tabular.tipo)

        X, y = self.preparador.transformar(filas)
        aleatorio = random.Random(self.cfg.semilla)
        indices = list(range(len(X)))
        aleatorio.shuffle(indices)
        corte = max(1, int(len(indices) * self.cfg.datos.val_proporcion))
        val_idx, train_idx = indices[:corte], indices[corte:]

        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

        def cargar(idx, mezclar):
            conjunto = TensorDataset(torch.tensor(X[idx]), torch.tensor(y[idx]))
            muestreador = None
            # Con clases desbalanceadas (ausencias, fraude, averías: siempre son pocas)
            # sin esto el modelo aprende a decir "no" y acierta el 87 % sin servir de nada.
            if mezclar and self.preparador.es_clasificacion and \
                    self.cfg.datos.balanceo == "sampler":
                conteo = np.bincount(y[idx].astype(int), minlength=len(self.preparador.clases))
                peso = {c: len(idx) / (len(conteo) * max(1, n)) for c, n in enumerate(conteo)}
                muestreador = WeightedRandomSampler(
                    [peso[int(v)] for v in y[idx]], num_samples=len(idx), replacement=True)
                mezclar = False
            return DataLoader(conjunto, batch_size=self.cfg.datos.batch, shuffle=mezclar,
                              sampler=muestreador, collate_fn=_juntar)

        print(f"tabular: {len(entradas)} columnas de entrada → «{objetivo}» "
              f"({'clasificación' if self.preparador.es_clasificacion else 'regresión'})"
              f" · {self.preparador.n_entradas} rasgos tras codificar")
        clases = self.preparador.clases or ["valor"]
        return cargar(train_idx, True), cargar(val_idx, False), InfoDatos(
            clases=clases, conteo=[0] * len(clases),
            n_train=len(train_idx), n_val=len(val_idx))

    def modelo(self, info: InfoDatos):
        salidas = len(self.preparador.clases) if self.preparador.es_clasificacion else 1
        capas, dentro = [], self.preparador.n_entradas
        for ancho in self.cfg.tabular.capas:
            capas += [nn.Linear(dentro, ancho), nn.BatchNorm1d(ancho), nn.ReLU(),
                      nn.Dropout(self.cfg.tabular.dropout)]
            dentro = ancho
        return nn.Sequential(*capas, nn.Linear(dentro, salidas))

    def criterio(self, info: InfoDatos, dispositivo):
        if self.preparador.es_clasificacion:
            return nn.CrossEntropyLoss(label_smoothing=self.cfg.perdida.suavizado).to(dispositivo)
        return nn.L1Loss().to(dispositivo)      # error absoluto: robusto ante casos raros

    def paso(self, modelo, lote, criterio, dispositivo, entrenando: bool,
             espejo: bool = False) -> Paso:
        x, y = lote
        x = x.to(dispositivo)
        y = y.to(dispositivo)
        salida = modelo(x)
        if self.preparador.es_clasificacion:
            # El objetivo viaja como entero: el evaluador lo usa para indexar la matriz
            y = y.long()
            perdida = criterio(salida, y)
        else:
            perdida = criterio(salida.squeeze(-1), y.float())
        return Paso(perdida=perdida, logits=salida.detach(), objetivos=y)

    def evaluador(self, info: InfoDatos):
        if self.preparador.es_clasificacion:
            from nucleo.metricas import Evaluador
            return Evaluador(info.clases, self.cfg.entrenamiento.metrica_objetivo)
        return EvaluadorRegresion(self.preparador)

    def descripcion(self) -> str:
        return f"red tabular {'→'.join(map(str, self.cfg.tabular.capas))}"

    def exportar_extra(self) -> dict:
        return {"arquitectura": "mlp_tabular", **self.preparador.estado()}


# ---------------------------------------------------------------- preparación

class Preparador:
    """Convierte el CSV en números: normaliza los numéricos y codifica los de texto."""

    def __init__(self, filas, entradas, objetivo, tipo):
        self.entradas, self.objetivo = entradas, objetivo
        valores_objetivo = [f[objetivo] for f in filas if f.get(objetivo) not in (None, "")]
        self.es_clasificacion = _es_categoria(valores_objetivo) if tipo == "auto" \
            else tipo == "clasificacion"
        self.clases = sorted(set(valores_objetivo)) if self.es_clasificacion else []

        self.numericas, self.categoricas = [], {}
        for columna in entradas:
            valores = [f.get(columna, "") for f in filas]
            if _es_numerica(valores):
                nums = [_numero(v) for v in valores if _numero(v) is not None]
                media = sum(nums) / max(1, len(nums))
                desv = (sum((n - media) ** 2 for n in nums) / max(1, len(nums))) ** 0.5 or 1.0
                self.numericas.append((columna, media, desv))
            else:
                self.categoricas[columna] = sorted({v for v in valores if v})[:50]

        self.n_entradas = len(self.numericas) + sum(len(v) for v in self.categoricas.values())

    def transformar(self, filas):
        X = np.zeros((len(filas), self.n_entradas), dtype=np.float32)
        y = np.zeros(len(filas), dtype=np.float32)
        indice_clase = {c: i for i, c in enumerate(self.clases)}

        for i, fila in enumerate(filas):
            posicion = 0
            for columna, media, desv in self.numericas:
                valor = _numero(fila.get(columna))
                X[i, posicion] = 0.0 if valor is None else (valor - media) / desv
                posicion += 1
            for columna, valores in self.categoricas.items():
                if fila.get(columna) in valores:      # codificación uno-de-N
                    X[i, posicion + valores.index(fila[columna])] = 1.0
                posicion += len(valores)
            bruto = fila.get(self.objetivo)
            y[i] = indice_clase.get(bruto, 0) if self.es_clasificacion else (_numero(bruto) or 0.0)
        return X, y

    def fila_a_vector(self, fila: dict) -> np.ndarray:
        return self.transformar([fila])[0]

    def estado(self) -> dict:
        return {"entradas": self.entradas, "objetivo": self.objetivo,
                "clases": self.clases, "es_clasificacion": self.es_clasificacion,
                "numericas": [(c, m, d) for c, m, d in self.numericas],
                "categoricas": self.categoricas, "n_entradas": self.n_entradas}


class EvaluadorRegresion:
    """Error absoluto medio y R²: cuánto se equivoca y cuánto explica de la variación."""

    def __init__(self, preparador):
        self.preparador = preparador
        self.metrica_objetivo = "acc"
        self.reiniciar()

    def reiniciar(self) -> None:
        self.predichos, self.reales = [], []
        self._perdida, self._n = 0.0, 0

    def actualizar(self, logits=None, objetivos=None, perdida=None, subgrupos=None,
                   datos_extra=None) -> None:
        if perdida is not None:
            self._perdida += perdida
            self._n += 1
        if logits is not None:
            self.predichos += logits.squeeze(-1).float().cpu().numpy().tolist()
            self.reales += objetivos.float().cpu().numpy().tolist()

    def resumen(self, calibrar: bool = False, curvas: bool = True) -> dict:
        if not self.reales:
            return {}
        p = np.array(self.predichos)
        r = np.array(self.reales)
        mae = float(np.abs(p - r).mean())
        varianza = float(((r - r.mean()) ** 2).mean()) or 1e-9
        r2 = 1 - float(((p - r) ** 2).mean()) / varianza
        return {"acc": max(0.0, r2), "acc_balanceada": max(0.0, r2), "mae": mae, "r2": r2,
                "perdida": self._perdida / max(1, self._n), "n": len(r),
                "texto": f"error medio {mae:.4f} · R² {r2:.4f} "
                         f"(R² 1,0 sería perfecto; 0 es no explicar nada)"}


def _juntar(lote):
    xs, ys = zip(*lote)
    return torch.stack(xs), torch.stack(ys)


def _leer_csv(ruta: Path):
    archivo = ruta / "datos.csv" if ruta.is_dir() else ruta
    if not archivo.exists():
        raise SystemExit(f"Falta {archivo} (un CSV con cabecera)")
    with archivo.open(encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    if not filas:
        raise SystemExit(f"{archivo} está vacío")
    return filas, list(filas[0])


def _numero(valor):
    try:
        n = float(str(valor).replace(",", "."))
        return None if math.isnan(n) else n
    except (TypeError, ValueError):
        return None


def _es_numerica(valores) -> bool:
    validos = [v for v in valores if v not in (None, "")]
    if not validos:
        return False
    return sum(_numero(v) is not None for v in validos) / len(validos) > 0.9


def _es_categoria(valores) -> bool:
    """Pocos valores distintos, o valores de texto: es una categoría."""
    distintos = set(valores)
    if not _es_numerica(valores):
        return True
    return len(distintos) <= max(2, min(20, len(valores) // 20))

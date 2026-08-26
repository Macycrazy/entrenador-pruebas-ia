"""Métricas de clasificación: acierto, matriz, AUC, PR, calibración y desglose por subgrupo.

Sin scikit-learn a propósito: son cuatro fórmulas y evita una dependencia pesada.
"""

from __future__ import annotations

import math

import numpy as np
import torch


class Evaluador:
    """Acumula logits y etiquetas de la validación y calcula todo al final.

    `subgrupos` es un dict {nombre: lista de valores por muestra} — por ejemplo
    {"etnia": [...], "edad": [...]} — y produce las mismas métricas por cada valor.
    """

    def __init__(self, clases: list[str], metrica_objetivo: str = "acc"):
        self.clases = clases
        self.metrica_objetivo = metrica_objetivo
        self.reiniciar()

    def reiniciar(self) -> None:
        self._logits: list[np.ndarray] = []
        self._objetivos: list[np.ndarray] = []
        self._subgrupos: dict[str, list] = {}
        self._perdida, self._n = 0.0, 0

    def actualizar(self, logits: torch.Tensor, objetivos: torch.Tensor,
                   perdida: float | None = None, subgrupos: dict | None = None,
                   datos_extra: dict | None = None) -> None:
        self._logits.append(logits.detach().float().cpu().numpy())
        self._objetivos.append(objetivos.detach().cpu().numpy())
        if perdida is not None:
            self._perdida += perdida * objetivos.size(0)
            self._n += objetivos.size(0)
        for nombre, valores in (subgrupos or {}).items():
            self._subgrupos.setdefault(nombre, []).extend(list(valores))

    # ------------------------------------------------------------------ cálculo

    def resumen(self, calibrar: bool = False, curvas: bool = True) -> dict:
        if not self._logits:
            return {}
        logits = np.concatenate(self._logits)
        objetivos = np.concatenate(self._objetivos)
        probs = _softmax(logits)

        salida = _basicas(probs, objetivos, self.clases)
        salida["perdida"] = self._perdida / max(1, self._n)
        salida["ece"] = _ece(probs, objetivos)

        if curvas and len(self.clases) == 2:
            salida["auc"] = _auc(probs[:, 1], (objetivos == 1).astype(int))
            salida["ap"] = _precision_media(probs[:, 1], (objetivos == 1).astype(int))
        elif curvas:
            salida["auc"] = float(np.mean([
                _auc(probs[:, i], (objetivos == i).astype(int)) for i in range(len(self.clases))
            ]))

        if calibrar:
            temperatura = ajustar_temperatura(logits, objetivos)
            probs_cal = _softmax(logits / temperatura)
            salida["temperatura"] = temperatura
            salida["ece_calibrado"] = _ece(probs_cal, objetivos)

        for nombre, valores in self._subgrupos.items():
            valores = np.array(valores[:len(objetivos)], dtype=object)
            salida.setdefault("subgrupos", {})[nombre] = {
                str(v): _basicas(probs[valores == v], objetivos[valores == v], self.clases)
                for v in sorted(set(valores.tolist()), key=str)
                if (valores == v).sum() >= 10
            }
        return salida

    @property
    def objetivo(self) -> str:
        return self.metrica_objetivo


def _basicas(probs: np.ndarray, objetivos: np.ndarray, clases: list[str]) -> dict:
    if len(objetivos) == 0:
        return {}
    predicho = probs.argmax(1)
    matriz = np.zeros((len(clases), len(clases)), dtype=int)
    for real, pred in zip(objetivos, predicho):
        matriz[real, pred] += 1

    por_clase = {}
    for i, clase in enumerate(clases):
        vp = matriz[i, i]
        precision = vp / max(1, matriz[:, i].sum())
        recall = vp / max(1, matriz[i].sum())
        por_clase[clase] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(2 * precision * recall / max(1e-9, precision + recall)),
            "n": int(matriz[i].sum()),
        }
    return {
        "acc": float((predicho == objetivos).mean()),
        "acc_balanceada": float(np.mean([v["recall"] for v in por_clase.values()])),
        "matriz": matriz.tolist(),
        "por_clase": por_clase,
        "n": int(len(objetivos)),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _auc(puntuacion: np.ndarray, positivo: np.ndarray) -> float:
    """Área bajo la curva ROC por el método de los rangos (equivale a Mann-Whitney U)."""
    n_pos, n_neg = int(positivo.sum()), int((1 - positivo).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    orden = np.argsort(puntuacion)
    rangos = np.empty(len(puntuacion), dtype=float)
    rangos[orden] = np.arange(1, len(puntuacion) + 1)
    # empates: rango medio
    valores, inversa, conteo = np.unique(puntuacion, return_inverse=True, return_counts=True)
    for i, c in enumerate(conteo):
        if c > 1:
            rangos[inversa == i] = rangos[inversa == i].mean()
    return float((rangos[positivo == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _precision_media(puntuacion: np.ndarray, positivo: np.ndarray) -> float:
    """Área bajo precisión-recall (average precision)."""
    orden = np.argsort(-puntuacion)
    positivo = positivo[orden]
    acumulado = np.cumsum(positivo)
    precision = acumulado / np.arange(1, len(positivo) + 1)
    total = positivo.sum()
    return float((precision * positivo).sum() / total) if total else float("nan")


def _ece(probs: np.ndarray, objetivos: np.ndarray, cajas: int = 15) -> float:
    """Error de calibración esperado: cuánto se despega la confianza del acierto real.

    Un ECE de 0,15 significa que cuando el modelo dice «90 %» acierta más bien el 75 %.
    """
    confianza = probs.max(1)
    acierto = (probs.argmax(1) == objetivos).astype(float)
    bordes = np.linspace(0, 1, cajas + 1)
    error = 0.0
    for i in range(cajas):
        dentro = (confianza > bordes[i]) & (confianza <= bordes[i + 1])
        if dentro.sum():
            error += dentro.mean() * abs(acierto[dentro].mean() - confianza[dentro].mean())
    return float(error)


def ajustar_temperatura(logits: np.ndarray, objetivos: np.ndarray) -> float:
    """Temperature scaling: divide los logits por T para que el porcentaje sea honesto."""
    t_logits = torch.tensor(logits, dtype=torch.float32)
    t_obj = torch.tensor(objetivos, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    # Sin búsqueda de línea, LBFGS se pasa de frenada y exp() desborda a inf/nan;
    # el clamp acota la temperatura a un rango sensato (0,05 - 20).
    optimizador = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50,
                                    line_search_fn="strong_wolfe")

    def paso():
        optimizador.zero_grad()
        perdida = torch.nn.functional.cross_entropy(
            t_logits / log_t.clamp(-3, 3).exp(), t_obj)
        perdida.backward()
        return perdida

    try:
        optimizador.step(paso)
    except RuntimeError:
        return 1.0
    temperatura = float(log_t.clamp(-3, 3).exp().item())
    return temperatura if math.isfinite(temperatura) and temperatura > 0 else 1.0


def formatear(resumen: dict, clases: list[str]) -> str:
    """Informe legible para el terminal."""
    if not resumen:
        return "(sin datos)"
    lineas = [f"acierto {resumen['acc']:.4f} · balanceado {resumen['acc_balanceada']:.4f}"
              f" · n={resumen['n']}"]
    if "auc" in resumen:
        lineas[0] += f" · AUC {resumen['auc']:.4f}"
    if "ece" in resumen:
        extra = f" → {resumen['ece_calibrado']:.4f} (T={resumen['temperatura']:.2f})" \
            if "ece_calibrado" in resumen else ""
        lineas[0] += f" · ECE {resumen['ece']:.4f}{extra}"

    # Tareas con miles de clases (identidades) no dan tabla por clase ni matriz:
    # aportan su propio bloque de texto (EER, TAR@FAR…).
    if not resumen.get("por_clase"):
        if resumen.get("texto"):
            lineas.append(resumen["texto"])
        return "\n".join(lineas)

    ancho = max(len(c) for c in clases) + 2
    lineas.append(f"{'clase':<{ancho}}{'precisión':>11}{'recall':>9}{'F1':>8}{'n':>8}")
    for clase in clases:
        m = resumen["por_clase"][clase]
        lineas.append(f"{clase:<{ancho}}{m['precision']:>11.4f}{m['recall']:>9.4f}"
                      f"{m['f1']:>8.4f}{m['n']:>8}")

    lineas.append("matriz (filas=real, columnas=predicho): " + str(resumen["matriz"]))

    for nombre, grupos in resumen.get("subgrupos", {}).items():
        lineas.append(f"\npor {nombre}:")
        lineas.append(f"  {'grupo':<22}{'acierto':>9}{'balanceado':>12}{'n':>8}")
        for grupo, m in sorted(grupos.items(), key=lambda kv: -kv[1]["n"]):
            lineas.append(f"  {grupo:<22}{m['acc']:>9.4f}{m['acc_balanceada']:>12.4f}{m['n']:>8}")
        peor = min(grupos.values(), key=lambda m: m["acc"], default=None)
        mejor = max(grupos.values(), key=lambda m: m["acc"], default=None)
        if peor and mejor:
            lineas.append(f"  brecha máxima: {mejor['acc'] - peor['acc']:.4f}")
    return "\n".join(lineas)

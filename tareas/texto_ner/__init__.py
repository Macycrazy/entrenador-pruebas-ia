"""Tarea: reconocimiento de entidades (NER) — encontrar nombres, lugares y cosas en un texto.

De fábrica el modelo base ya reconoce personas, lugares y organizaciones. Entrenarlo sirve
para entidades tuyas: gerencias, códigos de carnet, tipos de trámite, nombres de equipos.

Formato de datos, JSONL con una muestra por línea:

    {"texto": "Miguel Cárdenas trabaja en la Gerencia de Sistemas",
     "entidades": [[0, 15, "PER"], [30, 50, "ORG"]]}

La métrica es **F1 por entidad**: cuenta acierto solo si el tipo y los límites coinciden.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from nucleo.tarea import InfoDatos, Paso, Tarea, registrar


@registrar("texto_ner")
class TareaNER(Tarea):

    def datos(self):
        from transformers import AutoTokenizer

        muestras = _leer(Path(self.cfg.datos.ruta))
        tipos = sorted({e[2] for m in muestras for e in m.get("entidades", [])})
        self.etiquetas = ["O"] + [f"{prefijo}-{t}" for t in tipos for prefijo in ("B", "I")]
        self.indice = {e: i for i, e in enumerate(self.etiquetas)}

        self.tokenizador = AutoTokenizer.from_pretrained(self.cfg.ner.modelo_base)
        aleatorio = random.Random(self.cfg.semilla)
        aleatorio.shuffle(muestras)
        corte = max(1, int(len(muestras) * self.cfg.datos.val_proporcion))
        train, val = muestras[corte:], muestras[:corte]

        from torch.utils.data import DataLoader
        juntar = _crear_juntador(self.tokenizador)
        comunes = dict(num_workers=self.cfg.datos.workers, collate_fn=juntar)
        cargar = lambda datos, mezclar: DataLoader(  # noqa: E731
            DatasetNER(datos, self.tokenizador, self.indice, self.cfg),
            batch_size=self.cfg.datos.batch, shuffle=mezclar, **comunes)

        print(f"NER: {len(tipos)} tipos de entidad ({', '.join(tipos) or 'ninguno'})")
        return cargar(train, True), cargar(val, False), InfoDatos(
            clases=self.etiquetas, conteo=[0] * len(self.etiquetas),
            n_train=len(train), n_val=len(val))

    def modelo(self, info: InfoDatos):
        from transformers import AutoModelForTokenClassification
        return AutoModelForTokenClassification.from_pretrained(
            self.cfg.ner.modelo_base, num_labels=len(self.etiquetas),
            ignore_mismatched_sizes=True)

    def criterio(self, info: InfoDatos, dispositivo):
        return None      # el modelo calcula la pérdida con las etiquetas

    def paso(self, modelo, lote, criterio, dispositivo, entrenando: bool,
             espejo: bool = False) -> Paso:
        entradas, etiquetas, _, _ = lote
        entradas = {k: v.to(dispositivo) for k, v in entradas.items()}
        etiquetas = etiquetas.to(dispositivo)
        salida = modelo(**entradas, labels=etiquetas)
        return Paso(perdida=salida.loss, logits=salida.logits.detach(), objetivos=etiquetas)

    def evaluador(self, info: InfoDatos):
        return EvaluadorNER(self.etiquetas)

    def descripcion(self) -> str:
        return f"{self.cfg.ner.modelo_base} (NER)"

    def exportar_extra(self) -> dict:
        return {"modelo_base": self.cfg.ner.modelo_base, "etiquetas": self.etiquetas,
                "arquitectura": "ner"}


def _leer(ruta: Path) -> list[dict]:
    archivo = ruta / "entidades.jsonl" if ruta.is_dir() else ruta
    if not archivo.exists():
        raise SystemExit(
            f"Falta {archivo}. Una muestra por línea:\n"
            '  {"texto": "...", "entidades": [[inicio, fin, "TIPO"]]}')
    muestras = [json.loads(l) for l in archivo.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not muestras:
        raise SystemExit(f"{archivo} está vacío")
    return muestras


class DatasetNER(torch.utils.data.Dataset):
    def __init__(self, muestras, tokenizador, indice, cfg):
        self.muestras, self.tok, self.indice, self.cfg = muestras, tokenizador, indice, cfg

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, i):
        muestra = self.muestras[i]
        codificado = self.tok(muestra["texto"], truncation=True,
                              max_length=self.cfg.ner.longitud_max,
                              return_offsets_mapping=True)
        # Cada token recibe la etiqueta del carácter donde empieza
        etiquetas = []
        for inicio, fin in codificado["offset_mapping"]:
            if inicio == fin:
                etiquetas.append(-100)        # tokens especiales
                continue
            etiqueta = "O"
            for e_inicio, e_fin, tipo in muestra.get("entidades", []):
                if inicio >= e_inicio and fin <= e_fin:
                    etiqueta = f"{'B' if inicio == e_inicio else 'I'}-{tipo}"
                    break
            etiquetas.append(self.indice.get(etiqueta, 0))
        codificado.pop("offset_mapping")
        return {**codificado, "labels": etiquetas}


def _crear_juntador(tokenizador):
    def juntar(lote):
        largo = max(len(m["input_ids"]) for m in lote)
        relleno = tokenizador.pad_token_id or 0
        entradas = {
            "input_ids": torch.tensor([m["input_ids"] + [relleno] * (largo - len(m["input_ids"]))
                                       for m in lote]),
            "attention_mask": torch.tensor([m["attention_mask"] + [0] * (largo - len(m["attention_mask"]))
                                            for m in lote]),
        }
        etiquetas = torch.tensor([m["labels"] + [-100] * (largo - len(m["labels"])) for m in lote])
        return entradas, etiquetas, {}, {}
    return juntar


class EvaluadorNER:
    """F1 a nivel de entidad: solo cuenta si el tipo y los límites coinciden exactamente."""

    def __init__(self, etiquetas):
        self.etiquetas = etiquetas
        self.metrica_objetivo = "acc"
        self.reiniciar()

    def reiniciar(self) -> None:
        self.aciertos = self.predichas = self.reales = 0
        self._perdida, self._n = 0.0, 0

    def actualizar(self, logits=None, objetivos=None, perdida=None, subgrupos=None,
                   datos_extra=None) -> None:
        if perdida is not None:
            self._perdida += perdida
            self._n += 1
        if logits is None:
            return
        predicho = logits.argmax(-1).cpu().numpy()
        real = objetivos.cpu().numpy()
        for fila_p, fila_r in zip(predicho, real):
            validos = fila_r != -100
            a = _entidades([self.etiquetas[i] for i in fila_p[validos]])
            b = _entidades([self.etiquetas[i] for i in fila_r[validos]])
            self.aciertos += len(a & b)
            self.predichas += len(a)
            self.reales += len(b)

    def resumen(self, calibrar: bool = False, curvas: bool = True) -> dict:
        if not self._n:
            return {}
        precision = self.aciertos / max(1, self.predichas)
        recall = self.aciertos / max(1, self.reales)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        return {"acc": f1, "acc_balanceada": f1, "f1": f1, "precision": precision,
                "recall": recall, "perdida": self._perdida / self._n, "n": self.reales,
                "texto": f"F1 {f1:.4f} · precisión {precision:.4f} · recall {recall:.4f} "
                         f"({self.aciertos} de {self.reales} entidades)"}


def _entidades(etiquetas: list[str]) -> set:
    """Convierte la secuencia BIO en un conjunto de (inicio, fin, tipo)."""
    salida, actual = set(), None
    for i, etiqueta in enumerate(etiquetas):
        if etiqueta.startswith("B-"):
            if actual:
                salida.add(actual)
            actual = (i, i + 1, etiqueta[2:])
        elif etiqueta.startswith("I-") and actual and actual[2] == etiqueta[2:]:
            actual = (actual[0], i + 1, actual[2])
        else:
            if actual:
                salida.add(actual)
            actual = None
    if actual:
        salida.add(actual)
    return salida

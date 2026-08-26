"""Datos de instrucciones para ajustar un modelo de lenguaje.

JSONL, una muestra por línea, en cualquiera de estos dos formatos:

    {"instruccion": "Resume el reporte", "entrada": "…", "salida": "…"}
    {"texto": "cualquier texto para continuar entrenando el idioma"}

Con `llm.solo_respuesta` (por defecto), la pérdida se calcula **solo sobre la respuesta**:
el modelo aprende a responder, no a repetir la pregunta.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


def recopilar(cfg) -> tuple[list[dict], list[dict]]:
    raiz = Path(cfg.datos.ruta)
    if raiz.is_file():
        train, val = _leer_jsonl(raiz), None
    elif (raiz / "train.jsonl").exists():
        train = _leer_jsonl(raiz / "train.jsonl")
        val = _leer_jsonl(raiz / "val.jsonl") if (raiz / "val.jsonl").exists() else None
    else:
        raise SystemExit(
            f"No encuentro {raiz}/train.jsonl\n"
            "Para probar sin datos:  python preparacion/generar_instrucciones_sintetico.py")

    if val is None:
        aleatorio = random.Random(cfg.semilla)
        aleatorio.shuffle(train)
        corte = max(1, int(len(train) * cfg.datos.val_proporcion))
        train, val = train[corte:], train[:corte]

    if cfg.datos.limite:
        train = train[:cfg.datos.limite]
        val = val[:max(1, cfg.datos.limite // 5)]
    return train, val


def _leer_jsonl(ruta: Path) -> list[dict]:
    filas = []
    for numero, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), 1):
        linea = linea.strip()
        if not linea:
            continue
        try:
            filas.append(json.loads(linea))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{ruta}:{numero} no es JSON válido: {error}") from None
    if not filas:
        raise SystemExit(f"{ruta} está vacío")
    return filas


class DatasetInstrucciones(Dataset):
    def __init__(self, filas: list[dict], tokenizador, cfg):
        self.filas = filas
        self.tok = tokenizador
        self.cfg = cfg

    def __len__(self):
        return len(self.filas)

    def __getitem__(self, indice):
        fila = self.filas[indice]
        if "texto" in fila:
            ids = self.tok(fila["texto"], truncation=True,
                           max_length=self.cfg.llm.longitud_max)["input_ids"]
            return {"input_ids": ids, "labels": list(ids)}

        prompt = self.cfg.llm.plantilla.format(
            instruccion=fila.get("instruccion", ""), entrada=fila.get("entrada", "")).rstrip()
        respuesta = str(fila.get("salida", ""))

        ids_prompt = self.tok(prompt + "\n", add_special_tokens=False)["input_ids"]
        ids_respuesta = self.tok(respuesta + (self.tok.eos_token or ""),
                                 add_special_tokens=False)["input_ids"]
        ids = (ids_prompt + ids_respuesta)[:self.cfg.llm.longitud_max]

        if self.cfg.llm.solo_respuesta:
            # -100 = «no cuenta para la pérdida»
            etiquetas = ([-100] * min(len(ids_prompt), len(ids))
                         + ids[len(ids_prompt):])[:len(ids)]
        else:
            etiquetas = list(ids)
        return {"input_ids": ids, "labels": etiquetas}


def crear_juntador(tokenizador):
    relleno = tokenizador.pad_token_id or tokenizador.eos_token_id or 0

    def juntar(lote):
        largo = max(len(m["input_ids"]) for m in lote)
        ids, etiquetas, mascara = [], [], []
        for m in lote:
            falta = largo - len(m["input_ids"])
            ids.append(m["input_ids"] + [relleno] * falta)
            etiquetas.append(m["labels"] + [-100] * falta)
            mascara.append([1] * len(m["input_ids"]) + [0] * falta)
        entradas = {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mascara)}
        return entradas, torch.tensor(etiquetas), {}, {}
    return juntar


def crear_loaders(cfg, train, val, tokenizador):
    juntar = crear_juntador(tokenizador)
    comunes = dict(num_workers=cfg.datos.workers, collate_fn=juntar,
                   persistent_workers=cfg.datos.workers > 0)
    return (DataLoader(DatasetInstrucciones(train, tokenizador, cfg),
                       batch_size=cfg.datos.batch, shuffle=True, **comunes),
            DataLoader(DatasetInstrucciones(val, tokenizador, cfg),
                       batch_size=cfg.datos.batch, shuffle=False, **comunes))

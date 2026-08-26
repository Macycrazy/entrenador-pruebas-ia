"""Datos de texto: CSV con una columna de texto y otra de etiqueta, o carpetas por clase.

    datos_texto/train.csv   con cabecera:  texto,etiqueta
    datos_texto/val.csv

o bien:

    datos_texto/
        queja/     001.txt 002.txt …
        consulta/  …

Los nombres de las columnas se cambian con texto.columna_texto y texto.columna_etiqueta.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


def recopilar(cfg) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[str]]:
    raiz = Path(cfg.datos.ruta)
    if not raiz.exists():
        raise SystemExit(
            f"No existe {raiz}.\n"
            f"Esperado: {raiz}/train.csv con columnas "
            f"'{cfg.texto.columna_texto},{cfg.texto.columna_etiqueta}'\n"
            "Para probar sin datos:  python preparacion/generar_texto_sintetico.py")

    if (raiz / "train.csv").exists():
        train, clases = _desde_csv(raiz / "train.csv", cfg)
        val = (_desde_csv(raiz / "val.csv", cfg, clases)[0]
               if (raiz / "val.csv").exists() else None)
    elif raiz.suffix == ".csv" or (raiz / "datos.csv").exists():
        ruta = raiz if raiz.suffix == ".csv" else raiz / "datos.csv"
        train, clases = _desde_csv(ruta, cfg)
        val = None
    else:
        train, clases = _desde_carpetas(raiz, cfg)
        val = None

    if val is None:
        aleatorio = random.Random(cfg.semilla)
        aleatorio.shuffle(train)
        corte = max(1, int(len(train) * cfg.datos.val_proporcion))
        train, val = train[corte:], train[:corte]

    if cfg.datos.limite:
        train = train[:cfg.datos.limite]
        val = val[:max(1, cfg.datos.limite // 5)]
    return train, val, clases


def _desde_csv(ruta: Path, cfg, clases: list[str] | None = None):
    with ruta.open(encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    if not filas:
        raise SystemExit(f"{ruta} está vacío")

    col_texto, col_etiqueta = cfg.texto.columna_texto, cfg.texto.columna_etiqueta
    for columna in (col_texto, col_etiqueta):
        if columna not in filas[0]:
            raise SystemExit(f"{ruta} no tiene la columna '{columna}'. "
                             f"Tiene: {', '.join(filas[0])}")

    clases = clases or sorted({fila[col_etiqueta] for fila in filas})
    indice = {c: i for i, c in enumerate(clases)}
    muestras = [(fila[col_texto], indice[fila[col_etiqueta]])
                for fila in filas if fila[col_etiqueta] in indice and fila[col_texto].strip()]
    return muestras, clases


def _desde_carpetas(raiz: Path, cfg):
    carpetas = sorted(p.name for p in raiz.iterdir() if p.is_dir())
    clases = list(cfg.datos.clases) if cfg.datos.clases else carpetas
    muestras = []
    for i, clase in enumerate(clases):
        for archivo in sorted((raiz / clase).rglob("*.txt")):
            texto = archivo.read_text(encoding="utf-8", errors="ignore").strip()
            if texto:
                muestras.append((texto, i))
    if not muestras:
        raise SystemExit(f"No hay textos en {raiz}")
    return muestras, clases


class DatasetTexto(Dataset):
    def __init__(self, muestras, tokenizador, longitud_max: int):
        self.muestras = muestras
        self.tokenizador = tokenizador
        self.longitud_max = longitud_max

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, indice):
        texto, etiqueta = self.muestras[indice]
        codificado = self.tokenizador(texto, truncation=True, max_length=self.longitud_max)
        return codificado, etiqueta


def crear_juntador(tokenizador):
    """Relleno dinámico: cada lote se rellena a su texto más largo, no al máximo global."""
    def juntar(lote):
        codificados, etiquetas = zip(*lote)
        relleno = tokenizador.pad(list(codificados), return_tensors="pt")
        return relleno, torch.tensor(etiquetas, dtype=torch.long), {}, {}
    return juntar


def crear_loaders(cfg, train, val, tokenizador):
    ds_train = DatasetTexto(train, tokenizador, cfg.texto.longitud_max)
    ds_val = DatasetTexto(val, tokenizador, cfg.texto.longitud_max)
    juntar = crear_juntador(tokenizador)
    comunes = dict(num_workers=cfg.datos.workers, collate_fn=juntar,
                   persistent_workers=cfg.datos.workers > 0)
    return (DataLoader(ds_train, batch_size=cfg.datos.batch, shuffle=True, **comunes),
            DataLoader(ds_val, batch_size=cfg.datos.batch * 2, shuffle=False, **comunes),
            ds_train, ds_val)

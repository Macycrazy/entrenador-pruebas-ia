"""Datos para clasificación de imagen: splits, k-fold, subgrupos, deduplicado y muestreo."""

from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Muestra:
    ruta: Path
    etiqueta: int
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- recopilación

def recopilar(cfg) -> tuple[list[Muestra], list[Muestra], list[str]]:
    raiz = Path(cfg.datos.ruta)
    metadatos = _leer_metadatos(cfg)

    dir_train, dir_val = raiz / "train", raiz / "val"
    if cfg.datos.kfold:
        todas, clases = _listar(raiz if not dir_train.exists() else dir_train, cfg, metadatos)
        if dir_val.exists():
            extra, _ = _listar(dir_val, cfg, metadatos, clases)
            todas += extra
        train, val = _particion_kfold(todas, cfg)
    elif dir_train.exists() and dir_val.exists():
        train, clases = _listar(dir_train, cfg, metadatos)
        val, _ = _listar(dir_val, cfg, metadatos, clases)
    else:
        todas, clases = _listar(raiz, cfg, metadatos)
        train, val = _particion_simple(todas, cfg)

    if cfg.datos.deduplicar:
        antes = len(train) + len(val)
        train, val = _deduplicar(train, val)
        print(f"Deduplicado: {antes - len(train) - len(val)} imágenes repetidas fuera")

    if cfg.datos.limite:
        # Barajar antes de cortar: la lista viene ordenada por clase y quedarse con
        # las primeras N daría un subconjunto de una sola clase.
        aleatorio = random.Random(cfg.semilla)
        aleatorio.shuffle(train)
        aleatorio.shuffle(val)
        train = train[:cfg.datos.limite]
        val = val[:max(1, cfg.datos.limite // 5)]
    return train, val, clases


def _listar(raiz: Path, cfg, metadatos: dict, clases: list[str] | None = None):
    if not raiz.exists():
        raise SystemExit(f"No existe la carpeta de datos {raiz}")
    carpetas = sorted(p.name for p in raiz.iterdir() if p.is_dir())
    if not carpetas:
        raise SystemExit(f"{raiz} no contiene subcarpetas de clase")
    clases = clases or (cfg.datos.clases or carpetas)

    muestras = []
    for indice, clase in enumerate(clases):
        for ruta in sorted((raiz / clase).rglob("*")):
            if ruta.suffix.lower() in EXTENSIONES:
                muestras.append(Muestra(ruta, indice, metadatos.get(ruta.stem, {})))
    if not muestras:
        raise SystemExit(f"No se encontraron imágenes en {raiz}")
    return muestras, list(clases)


def _leer_metadatos(cfg) -> dict[str, dict]:
    """CSV opcional con columnas extra por imagen (edad, etnia, identidad…)."""
    ruta = cfg.datos.metadatos
    if not ruta:
        return {}
    ruta = Path(ruta)
    if not ruta.exists():
        raise SystemExit(f"No existe el CSV de metadatos {ruta}")
    with ruta.open() as f:
        filas = list(csv.DictReader(f))
    if not filas:
        return {}
    clave = "archivo" if "archivo" in filas[0] else list(filas[0])[0]
    # Se indexa por nombre sin extensión: preparar_datos.py reescribe los recortes
    # como .jpg y el nombre completo dejaría de casar.
    return {Path(fila[clave]).stem: fila for fila in filas}


# ---------------------------------------------------------------- particiones

def _grupo(muestra: Muestra, columna: str | None) -> str:
    """Clave de agrupación: evita que la misma persona caiga en train y en val."""
    if columna and columna in muestra.meta:
        return str(muestra.meta[columna])
    return str(muestra.ruta)


def _particion_simple(muestras: list[Muestra], cfg):
    aleatorio = random.Random(cfg.semilla)
    columna = cfg.datos.agrupar_por
    por_grupo = defaultdict(list)
    for m in muestras:
        por_grupo[_grupo(m, columna)].append(m)

    grupos = list(por_grupo)
    aleatorio.shuffle(grupos)
    corte = max(1, int(len(grupos) * cfg.datos.val_proporcion))
    val_grupos = set(grupos[:corte])
    train = [m for g, ms in por_grupo.items() if g not in val_grupos for m in ms]
    val = [m for g in val_grupos for m in por_grupo[g]]
    return train, val


def _particion_kfold(muestras: list[Muestra], cfg):
    k, fold = cfg.datos.kfold, cfg.datos.fold
    if not 0 <= fold < k:
        raise SystemExit(f"fold debe estar entre 0 y {k - 1}")
    aleatorio = random.Random(cfg.semilla)
    columna = cfg.datos.agrupar_por
    por_grupo = defaultdict(list)
    for m in muestras:
        por_grupo[_grupo(m, columna)].append(m)

    grupos = sorted(por_grupo)
    aleatorio.shuffle(grupos)
    val_grupos = set(grupos[fold::k])
    train = [m for g in grupos if g not in val_grupos for m in por_grupo[g]]
    val = [m for g in val_grupos for m in por_grupo[g]]
    print(f"k-fold {fold + 1}/{k}: {len(train)} train · {len(val)} val")
    return train, val


def _hash_perceptual(ruta: Path) -> str:
    try:
        with Image.open(ruta) as imagen:
            gris = imagen.convert("L").resize((8, 8), Image.BILINEAR)
    except Exception:  # noqa: BLE001 - una imagen ilegible no debe tumbar el proceso
        return hashlib.md5(str(ruta).encode()).hexdigest()
    pixeles = list(gris.getdata())
    media = sum(pixeles) / len(pixeles)
    return "".join("1" if p > media else "0" for p in pixeles)


def _deduplicar(train: list[Muestra], val: list[Muestra]):
    vistos: set[str] = set()
    salida = []
    for conjunto in (val, train):        # val primero: se prioriza mantener validación intacta
        limpio = []
        for m in conjunto:
            h = _hash_perceptual(m.ruta)
            if h in vistos:
                continue
            vistos.add(h)
            limpio.append(m)
        salida.append(limpio)
    return salida[1], salida[0]


# ---------------------------------------------------------------- dataset

class DatasetImagenes(Dataset):
    def __init__(self, muestras: list[Muestra], transformacion, subgrupos: list[str],
                 extras: dict[str, dict], cache: bool = False):
        self.muestras = muestras
        self.transformacion = transformacion
        self.subgrupos = subgrupos
        self.extras = extras          # {columna: {valor: índice}}
        self.cache: dict[int, bytes] = {} if cache else None

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, indice):
        muestra = self.muestras[indice]
        if self.cache is not None:
            datos = self.cache.get(indice)
            if datos is None:
                datos = self.cache[indice] = muestra.ruta.read_bytes()
            import io
            imagen = Image.open(io.BytesIO(datos)).convert("RGB")
        else:
            with Image.open(muestra.ruta) as bruta:
                imagen = bruta.convert("RGB")

        x = self.transformacion(imagen)
        meta = {clave: muestra.meta.get(clave, "?") for clave in self.subgrupos}
        extras = {columna: vocab.get(muestra.meta.get(columna), -100)
                  for columna, vocab in self.extras.items()}
        return x, muestra.etiqueta, meta, extras

    def cambiar_transformacion(self, transformacion) -> None:
        self.transformacion = transformacion


def juntar(lote):
    """Collate propio: apila imágenes y etiquetas, y agrupa metadatos en listas."""
    xs, ys, metas, extras = zip(*lote)
    meta_agrupada = {clave: [m[clave] for m in metas] for clave in (metas[0] or {})}
    extra_agrupada = {clave: torch.tensor([e[clave] for e in extras], dtype=torch.long)
                      for clave in (extras[0] or {})}
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long), meta_agrupada, extra_agrupada


# ---------------------------------------------------------------- loaders

def vocabularios(muestras: list[Muestra], columnas: list[str]) -> dict[str, dict]:
    vocab = {}
    for columna in columnas:
        valores = sorted({m.meta[columna] for m in muestras if columna in m.meta})
        vocab[columna] = {valor: i for i, valor in enumerate(valores)}
    return vocab


def crear_loaders(cfg, train: list[Muestra], val: list[Muestra], trans_train, trans_val,
                  extras: dict[str, dict]):
    d = cfg.datos
    ds_train = DatasetImagenes(train, trans_train, d.subgrupos, extras, d.cache_ram)
    ds_val = DatasetImagenes(val, trans_val, d.subgrupos, extras, d.cache_ram)

    comunes = dict(num_workers=d.workers, pin_memory=torch.cuda.is_available(),
                   persistent_workers=d.workers > 0, collate_fn=juntar)
    if d.workers > 0:
        comunes["prefetch_factor"] = 4

    muestreador = None
    if d.balanceo == "sampler":
        conteo = Counter(m.etiqueta for m in train)
        pesos_clase = {c: len(train) / (len(conteo) * n) for c, n in conteo.items()}
        pesos = [pesos_clase[m.etiqueta] for m in train]
        muestreador = WeightedRandomSampler(pesos, num_samples=len(train), replacement=True)

    loader_train = DataLoader(ds_train, batch_size=d.batch, shuffle=muestreador is None,
                              sampler=muestreador, drop_last=len(train) > d.batch, **comunes)
    loader_val = DataLoader(ds_val, batch_size=d.batch * 2, shuffle=False, **comunes)
    return loader_train, loader_val, ds_train, ds_val

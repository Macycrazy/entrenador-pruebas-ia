"""Descarga imágenes de rostros etiquetadas por género desde datasets públicos.

Usa los parquet que Hugging Face sirve para cada dataset (no hace falta cuenta ni
token) y escribe las imágenes ya clasificadas en datos/crudo/{hombre,mujer}.

    python entrenamiento/descargar_dataset.py --listar
    python entrenamiento/descargar_dataset.py --fuente fairface --max-por-clase 12000
    python entrenamiento/descargar_dataset.py --fuente todas --max-por-clase 8000

Después:
    python entrenamiento/preparar_datos.py --recortar --val 0.15
    python entrenamiento/entrenar.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from comun.etiquetas import CLASES

API = "https://huggingface.co/api/datasets"
FIRMAS = {b"\xff\xd8\xff": ".jpg", b"\x89PNG": ".png", b"RIFF": ".webp", b"BM": ".bmp"}


MAPA_EDAD = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
MAPA_ETNIA = ["asiatico_oriental", "indio", "negro", "blanco", "medio_oriente",
              "latino_hispano", "asiatico_sudeste"]


@dataclass(frozen=True)
class Fuente:
    id: str
    config: str
    splits: tuple[str, ...]
    columna: str
    mapa: dict
    filas: int
    gb: float
    nota: str
    # columna del parquet -> (nombre en el CSV, lista de valores o None si ya es texto)
    extras: dict = None


FUENTES: dict[str, Fuente] = {
    "fairface": Fuente(
        # config 0.25 = recorte ajustado al rostro etiquetado. La variante "1.25" trae
        # mucho fondo y a veces otras caras, que confundirían al recorte automático.
        id="HuggingFaceM4/FairFace", config="0.25", splits=("train", "validation"),
        columna="gender", mapa={0: "hombre", 1: "mujer"}, filas=97698, gb=0.56,
        nota="Equilibrado por etnia y edad. La mejor opción para reducir sesgos.",
        extras={"age": ("edad", MAPA_EDAD), "race": ("etnia", MAPA_ETNIA)},
    ),
    "utkface": Fuente(
        id="deedax/UTK-Face-Revised", config="default", splits=("train", "valid"),
        columna="gender", mapa={"Male": "hombre", "Female": "mujer"}, filas=8469, gb=0.39,
        nota="UTKFace (subconjunto revisado). Rostros ya recortados, mucha variedad de edad.",
        extras={"age_group": ("edad", None), "race": ("etnia", None)},
    ),
    "celeba": Fuente(
        id="tpremoli/CelebA-attrs", config="default", splits=("train", "validation", "test"),
        columna="Male", mapa={1: "hombre", -1: "mujer"}, filas=202599, gb=1.42,
        nota="CelebA: caras de famosos, muy maquilladas y poco variadas en edad.",
    ),
    "gender5k": Fuente(
        id="myvision/gender-classification", config="default", splits=("train", "test", "eval"),
        columna="label", mapa={1: "hombre", 0: "mujer"}, filas=7000, gb=0.16,
        nota="Pequeño y rápido. Útil para una primera prueba de todo el circuito.",
    ),
}

ORDEN_RECOMENDADO = ["fairface", "utkface", "gender5k", "celeba"]


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Descarga datasets de rostros etiquetados")
    p.add_argument("--fuente", default="fairface",
                   choices=[*FUENTES, "todas"], help="Dataset a descargar")
    p.add_argument("--destino", type=Path, default=RAIZ / "datos" / "crudo")
    p.add_argument("--cache", type=Path, default=RAIZ / "datos" / ".cache_parquet")
    p.add_argument("--max-por-clase", type=int, default=10000,
                   help="Máximo de imágenes por clase y fuente (0 = todas)")
    p.add_argument("--splits", nargs="*", help="Splits concretos (por defecto, todos)")
    p.add_argument("--config", help="Config alternativa del dataset (fairface: 0.25 o 1.25)")
    p.add_argument("--conservar-parquet", action="store_true",
                   help="No borrar los .parquet descargados (ocupan bastante)")
    p.add_argument("--listar", action="store_true", help="Mostrar las fuentes disponibles y salir")
    p.add_argument("--solo-metadatos", action="store_true",
                   help="Regenerar metadatos.csv sin volver a escribir las imágenes "
                        "(el orden de extracción es determinista, así que los nombres coinciden)")
    return p.parse_args()


def main() -> None:
    args = argumentos()
    if args.listar:
        listar()
        return

    nombres = ORDEN_RECOMENDADO if args.fuente == "todas" else [args.fuente]
    for clase in CLASES:
        (args.destino / clase).mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    total = Counter()
    for nombre in nombres:
        try:
            conteo = descargar_fuente(FUENTES[nombre], nombre, args)
        except KeyboardInterrupt:
            print("\nInterrumpido. Las imágenes ya escritas siguen siendo válidas.")
            break
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"  ERROR de red en '{nombre}': {error}. Se continúa con la siguiente fuente.")
            continue
        total.update(conteo)

    print("\n" + "=" * 62)
    for clase in CLASES:
        existentes = len(list((args.destino / clase).glob("*")))
        print(f"  {clase:<8} +{total[clase]:>6} nuevas   ({existentes} en total en disco)")
    print("=" * 62)
    print(f"\nCarpeta: {args.destino}")
    print("Siguiente paso:  python entrenamiento/preparar_datos.py --recortar --val 0.15")


def listar() -> None:
    print(f"{'clave':<12}{'dataset':<34}{'filas':>9}{'tamaño':>9}")
    print("-" * 64)
    for clave in ORDEN_RECOMENDADO:
        f = FUENTES[clave]
        print(f"{clave:<12}{f.id:<34}{f.filas:>9}{f.gb:>7.2f} GB")
        print(f"{'':<12}{f.nota}")
    print("\nSe descargan los .parquet oficiales de Hugging Face; el caché temporal se borra")
    print("al terminar salvo que uses --conservar-parquet.")


def descargar_fuente(fuente: Fuente, nombre: str, args: argparse.Namespace) -> Counter:
    config = args.config or fuente.config
    print(f"\n### {nombre} · {fuente.id} ({config})")
    listado = pedir_json(f"{API}/{fuente.id}/parquet")
    if config not in listado:
        raise SystemExit(f"El dataset no expone la config '{config}'. "
                         f"Disponibles: {list(listado)}")

    splits = args.splits or fuente.splits
    cuota = args.max_por_clase or None
    conteo: Counter = Counter()
    # Al regenerar metadatos hay que empezar en 1 para reproducir la numeración original.
    indice = 1 if args.solo_metadatos else siguiente_indice(args.destino, nombre)
    filas_meta: list[dict] = []
    if args.solo_metadatos:
        (args.destino / "metadatos.csv").unlink(missing_ok=True)

    for split in splits:
        urls = listado[config].get(split)
        if not urls:
            print(f"  split '{split}' no disponible, se omite")
            continue
        for url in urls:
            if cuota and all(conteo[c] >= cuota for c in CLASES):
                print("  cuota alcanzada, no se descargan más archivos")
                _guardar_metadatos(args.destino, filas_meta)
                return conteo

            archivo = args.cache / f"{nombre}_{split}_{Path(url).name}"
            descargar(url, archivo)
            indice = extraer(archivo, fuente, nombre, args.destino, cuota, conteo,
                             indice, filas_meta, args.solo_metadatos)
            if not args.conservar_parquet:
                archivo.unlink(missing_ok=True)
            print(f"  acumulado: " + " · ".join(f"{c} {conteo[c]}" for c in CLASES))

    _guardar_metadatos(args.destino, filas_meta)
    return conteo


def _guardar_metadatos(destino: Path, filas: list[dict]) -> None:
    """CSV con las etiquetas extra (edad, etnia): habilita métricas por subgrupo y multitarea."""
    if not filas:
        return
    ruta = destino / "metadatos.csv"
    existentes = ruta.exists()
    campos = list(filas[0])
    with ruta.open("a", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        if not existentes:
            escritor.writeheader()
        escritor.writerows(filas)
    print(f"  metadatos: +{len(filas)} filas en {ruta}")


def pedir_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as respuesta:
        return json.load(respuesta)


def descargar(url: str, destino: Path) -> None:
    if destino.exists() and destino.stat().st_size > 0:
        print(f"  {destino.name}: ya en caché")
        return

    parcial = destino.with_suffix(destino.suffix + ".parcial")
    with urllib.request.urlopen(url, timeout=120) as respuesta:
        total = int(respuesta.headers.get("Content-Length") or 0)
        leidos = 0
        with parcial.open("wb") as salida:
            while trozo := respuesta.read(1 << 20):
                salida.write(trozo)
                leidos += len(trozo)
                pct = f" ({leidos / total * 100:.0f}%)" if total else ""
                print(f"  {destino.name}: {leidos / 1e6:.0f} MB{pct}", end="\r", flush=True)
    parcial.rename(destino)
    print(f"  {destino.name}: {leidos / 1e6:.0f} MB descargados" + " " * 15)


def extraer(archivo: Path, fuente: Fuente, prefijo: str, destino: Path,
            cuota: int | None, conteo: Counter, indice: int, filas_meta: list,
            solo_metadatos: bool = False) -> int:
    import pyarrow.parquet as pq

    lector = pq.ParquetFile(archivo)
    disponibles = set(lector.schema_arrow.names)
    columnas = ["image", fuente.columna]
    faltantes = [c for c in columnas if c not in disponibles]
    if faltantes:
        raise SystemExit(f"El parquet no tiene las columnas {faltantes}; "
                         f"tiene {sorted(disponibles)}")
    extras = {col: destino_col for col, destino_col in (fuente.extras or {}).items()
              if col in disponibles}
    columnas += list(extras)

    for lote in lector.iter_batches(batch_size=256, columns=columnas):
        for fila in lote.to_pylist():
            clase = fuente.mapa.get(fila[fuente.columna])
            if clase is None or (cuota and conteo[clase] >= cuota):
                continue
            datos = (fila.get("image") or {}).get("bytes")
            if not datos:
                continue
            nombre = f"{prefijo}_{indice:07d}"
            if not solo_metadatos:
                (destino / clase / f"{nombre}{extension(datos)}").write_bytes(datos)

            if extras:
                meta = {"archivo": nombre, "clase": clase}
                for columna, (etiqueta, mapa) in extras.items():
                    meta[etiqueta] = _valor(fila.get(columna), mapa)
                filas_meta.append(meta)

            conteo[clase] += 1
            indice += 1
            if conteo[clase] % 500 == 0:
                print(f"  extrayendo… {clase} {conteo[clase]}", end="\r", flush=True)
        if cuota and all(conteo[c] >= cuota for c in CLASES):
            break
    return indice


def _valor(bruto, mapa: list | None) -> str:
    """Traduce el código numérico del parquet a texto legible (para los subgrupos)."""
    if bruto is None:
        return "?"
    if mapa and isinstance(bruto, int) and 0 <= bruto < len(mapa):
        return mapa[bruto]
    return str(bruto)


def extension(datos: bytes) -> str:
    for firma, ext in FIRMAS.items():
        if datos.startswith(firma):
            return ext
    return ".jpg"


def siguiente_indice(destino: Path, prefijo: str) -> int:
    """Continúa la numeración para no pisar imágenes de descargas anteriores."""
    maximo = 0
    for clase in CLASES:
        for ruta in (destino / clase).glob(f"{prefijo}_*"):
            numero = ruta.stem.split("_")[-1]
            if numero.isdigit():
                maximo = max(maximo, int(numero))
    return maximo + 1


if __name__ == "__main__":
    main()

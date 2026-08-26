"""Descarga datasets de rostros **por identidad** (para reconocimiento facial).

A diferencia de descargar_dataset.py, aquí cada carpeta es una persona, no una clase:

    datos_rostros/
        Aaron_Peirsol/  0001.jpg 0002.jpg …
        Abdullah_Gul/   0001.jpg …

    python preparacion/descargar_rostros.py --listar
    python preparacion/descargar_rostros.py --fuente lfw
    python preparacion/descargar_rostros.py --fuente lfw_pares     # protocolo de verificación
    python preparacion/descargar_rostros.py --fuente casia --max-identidades 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from descargar_dataset import FIRMAS, descargar, extension, pedir_json  # noqa: E402

API = "https://huggingface.co/api/datasets"
INFO = "https://datasets-server.huggingface.co/info"


@dataclass(frozen=True)
class FuenteRostros:
    id: str
    config: str
    splits: tuple[str, ...]
    columna_etiqueta: str
    filas: int
    gb: float
    nota: str
    pares: bool = False


FUENTES = {
    "lfw": FuenteRostros(
        id="logasja/lfw", config="default", splits=("train",), columna_etiqueta="label",
        filas=13233, gb=0.24,
        nota="Labeled Faces in the Wild: 5 749 personas. Pequeño; ideal para probar.",
    ),
    "lfw_pares": FuenteRostros(
        id="logasja/lfw", config="pairs", splits=("test", "train"), columna_etiqueta="pair",
        filas=3200, gb=0.10, pares=True,
        nota="Protocolo oficial de verificación de LFW: pares etiquetados igual/distinto.",
    ),
    "casia": FuenteRostros(
        id="SaffalPoosh/casia_web_face", config="default", splits=("train",),
        columna_etiqueta="label", filas=490592, gb=9.53,
        nota="CASIA-WebFace: 10 575 identidades. El conjunto clásico de entrenamiento.",
    ),
    "vggface2": FuenteRostros(
        id="chronopt-research/cropped-vggface2-224", config="default",
        splits=("train", "validation"), columna_etiqueta="label", filas=3308040, gb=20.3,
        nota="VGGFace2 recortado a 224px: 8 631 identidades, 3,3 M de fotos. El mejor y el mayor.",
    ),
}


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Descarga datasets de rostros por identidad")
    p.add_argument("--fuente", default="lfw", choices=[*FUENTES])
    p.add_argument("--destino", type=Path, default=RAIZ / "datos_rostros")
    p.add_argument("--cache", type=Path, default=RAIZ / "datos" / ".cache_parquet")
    p.add_argument("--max-identidades", type=int, default=0, help="0 = todas")
    p.add_argument("--max-por-identidad", type=int, default=0, help="0 = todas")
    p.add_argument("--min-por-identidad", type=int, default=1,
                   help="Descarta identidades con menos fotos (2 o más para entrenar)")
    p.add_argument("--conservar-parquet", action="store_true")
    p.add_argument("--listar", action="store_true")
    return p.parse_args()


def main() -> None:
    args = argumentos()
    if args.listar:
        print(f"{'clave':<12}{'dataset':<44}{'filas':>10}{'tamaño':>9}")
        print("-" * 76)
        for clave, f in FUENTES.items():
            print(f"{clave:<12}{f.id:<44}{f.filas:>10}{f.gb:>7.2f} GB")
            print(f"{'':<12}{f.nota}")
        return

    fuente = FUENTES[args.fuente]
    args.destino.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    if fuente.pares:
        descargar_pares(fuente, args)
    else:
        descargar_identidades(fuente, args)


# ---------------------------------------------------------------- identidades

def descargar_identidades(fuente: FuenteRostros, args) -> None:
    import pyarrow.parquet as pq

    nombres = nombres_de_clase(fuente)
    print(f"### {args.fuente} · {fuente.id} · {len(nombres)} identidades")
    listado = pedir_json(f"{API}/{fuente.id}/parquet")[fuente.config]

    conteo: Counter = Counter()
    identidades: set[int] = set()

    for split in fuente.splits:
        for url in listado.get(split, []):
            if args.max_identidades and len(identidades) >= args.max_identidades \
                    and _completas(conteo, args):
                print("  cuota alcanzada")
                break
            archivo = args.cache / f"{args.fuente}_{split}_{Path(url).name}"
            descargar(url, archivo)

            lector = pq.ParquetFile(archivo)
            for lote in lector.iter_batches(batch_size=256,
                                            columns=["image", fuente.columna_etiqueta]):
                for fila in lote.to_pylist():
                    etiqueta = fila[fuente.columna_etiqueta]
                    if args.max_identidades and etiqueta not in identidades \
                            and len(identidades) >= args.max_identidades:
                        continue
                    if args.max_por_identidad and conteo[etiqueta] >= args.max_por_identidad:
                        continue
                    datos = (fila.get("image") or {}).get("bytes")
                    if not datos:
                        continue
                    nombre = _limpiar(nombres[etiqueta] if etiqueta < len(nombres)
                                      else str(etiqueta))
                    carpeta = args.destino / nombre
                    carpeta.mkdir(exist_ok=True)
                    conteo[etiqueta] += 1
                    identidades.add(etiqueta)
                    (carpeta / f"{conteo[etiqueta]:04d}{extension(datos)}").write_bytes(datos)
                    if sum(conteo.values()) % 1000 == 0:
                        print(f"  {sum(conteo.values())} fotos · "
                              f"{len(identidades)} identidades", end="\r", flush=True)
            if not args.conservar_parquet:
                archivo.unlink(missing_ok=True)

    descartadas = _podar(args.destino, args.min_por_identidad)
    total = sum(conteo.values())
    print(f"\n{total} fotos · {len(identidades) - descartadas} identidades en {args.destino}"
          + (f" ({descartadas} descartadas por tener menos de "
             f"{args.min_por_identidad} fotos)" if descartadas else ""))
    print("\nSiguiente paso:  python entrenar.py --config configs/rostro_id.yaml")


def _completas(conteo: Counter, args) -> bool:
    return bool(args.max_por_identidad) and all(
        n >= args.max_por_identidad for n in conteo.values())


def _podar(destino: Path, minimo: int) -> int:
    """Quita identidades con muy pocas fotos: no aportan y estropean las métricas."""
    if minimo <= 1:
        return 0
    quitadas = 0
    for carpeta in destino.iterdir():
        if carpeta.is_dir() and len(list(carpeta.glob("*"))) < minimo:
            for archivo in carpeta.iterdir():
                archivo.unlink()
            carpeta.rmdir()
            quitadas += 1
    return quitadas


def _limpiar(nombre: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(nombre))[:80]


def nombres_de_clase(fuente: FuenteRostros) -> list[str]:
    """Los nombres de las identidades están en los metadatos, no en el parquet."""
    url = f"{INFO}?dataset={fuente.id.replace('/', '%2F')}&config={fuente.config}"
    with urllib.request.urlopen(url, timeout=60) as respuesta:
        info = json.load(respuesta)
    caracteristicas = info["dataset_info"]["features"]
    return caracteristicas.get(fuente.columna_etiqueta, {}).get("names", [])


# ---------------------------------------------------------------- pares

def descargar_pares(fuente: FuenteRostros, args) -> None:
    """Guarda los pares de verificación: dos fotos y si son o no la misma persona."""
    import pyarrow.parquet as pq

    destino = args.destino.parent / "datos_rostros_pares"
    destino.mkdir(parents=True, exist_ok=True)
    listado = pedir_json(f"{API}/{fuente.id}/parquet")[fuente.config]
    filas: list[dict] = []
    indice = 0

    for split in fuente.splits:
        for url in listado.get(split, []):
            archivo = args.cache / f"pares_{split}_{Path(url).name}"
            descargar(url, archivo)
            lector = pq.ParquetFile(archivo)
            for lote in lector.iter_batches(batch_size=128,
                                            columns=["img_0", "img_1", "pair"]):
                for fila in lote.to_pylist():
                    a = (fila.get("img_0") or {}).get("bytes")
                    b = (fila.get("img_1") or {}).get("bytes")
                    if not a or not b:
                        continue
                    indice += 1
                    ruta_a = destino / f"{indice:05d}_a{extension(a)}"
                    ruta_b = destino / f"{indice:05d}_b{extension(b)}"
                    ruta_a.write_bytes(a)
                    ruta_b.write_bytes(b)
                    filas.append({"a": ruta_a.name, "b": ruta_b.name,
                                  "misma_persona": int(fila["pair"]), "split": split})
            if not args.conservar_parquet:
                archivo.unlink(missing_ok=True)

    csv_ruta = destino / "pares.csv"
    with csv_ruta.open("w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=["a", "b", "misma_persona", "split"])
        escritor.writeheader()
        escritor.writerows(filas)

    iguales = sum(f["misma_persona"] for f in filas)
    print(f"\n{len(filas)} pares ({iguales} de la misma persona, {len(filas) - iguales} distintas)")
    print(f"Índice en {csv_ruta}")
    print("\nÚsalo con:  python evaluar_rostros.py experimentos/rostro_id "
          f"--pares {csv_ruta}")


if __name__ == "__main__":
    main()

"""Importa el dataset UTKFace a datos/crudo/{hombre,mujer}.

UTKFace codifica las etiquetas en el nombre del archivo:
    edad_genero_etnia_fecha.jpg      con genero: 0 = hombre, 1 = mujer

    python entrenamiento/importar_utkface.py --origen ~/Descargas/UTKFace

Sirve igual para cualquier dataset con ese formato de nombre. Para otros datasets
basta con dejar las imágenes a mano en datos/crudo/hombre y datos/crudo/mujer.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from comun.etiquetas import CLASES

EXTENSIONES = {".jpg", ".jpeg", ".png"}
GENERO = {"0": "hombre", "1": "mujer"}


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Importa UTKFace a datos/crudo")
    p.add_argument("--origen", type=Path, required=True, help="Carpeta con las imágenes UTKFace")
    p.add_argument("--destino", type=Path, default=RAIZ / "datos" / "crudo")
    p.add_argument("--enlaces", action="store_true",
                   help="Crear enlaces simbólicos en vez de copiar (no duplica espacio)")
    p.add_argument("--max-por-clase", type=int, default=0, help="0 = sin límite")
    return p.parse_args()


def main() -> None:
    args = argumentos()
    if not args.origen.is_dir():
        raise SystemExit(f"No existe la carpeta {args.origen}")

    for clase in CLASES:
        (args.destino / clase).mkdir(parents=True, exist_ok=True)

    conteo = {clase: 0 for clase in CLASES}
    invalidas = 0

    for imagen in sorted(args.origen.rglob("*")):
        if imagen.suffix.lower() not in EXTENSIONES:
            continue
        partes = imagen.stem.split("_")
        if len(partes) < 3 or partes[1] not in GENERO:
            invalidas += 1
            continue
        clase = GENERO[partes[1]]
        if args.max_por_clase and conteo[clase] >= args.max_por_clase:
            continue

        salida = args.destino / clase / imagen.name
        if not salida.exists():
            if args.enlaces:
                salida.symlink_to(imagen.resolve())
            else:
                shutil.copy2(imagen, salida)
        conteo[clase] += 1

    for clase, n in conteo.items():
        print(f"{clase}: {n} imágenes")
    if invalidas:
        print(f"{invalidas} archivos ignorados (nombre sin etiqueta reconocible)")
    print("\nAhora: python entrenamiento/preparar_datos.py --recortar")


if __name__ == "__main__":
    main()

"""Prepara el dataset: reparte datos/crudo/{hombre,mujer} en train/ y val/.

Opcionalmente recorta el rostro de cada imagen (--recortar), que es lo que hace
el servidor en tiempo real: entrenar con recortes hace que ambas etapas vean lo mismo.

    python entrenamiento/preparar_datos.py --recortar --val 0.15
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import cv2

from comun.etiquetas import CLASES
from comun.rostros import DetectorRostros, recortar_rostro

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Divide el dataset en train/val")
    p.add_argument("--crudo", type=Path, default=RAIZ / "datos" / "crudo")
    p.add_argument("--destino", type=Path, default=RAIZ / "datos")
    p.add_argument("--val", type=float, default=0.15, help="Proporción para validación")
    p.add_argument("--recortar", action="store_true", help="Recortar el rostro de cada imagen")
    p.add_argument("--tam-min", type=int, default=64, help="Descartar rostros más pequeños (px)")
    p.add_argument("--sin-rostro", default="usar", choices=["usar", "descartar"],
                   help="Qué hacer si no se detecta rostro: usar la imagen entera "
                        "(los datasets ya vienen recortados) o descartarla")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--limpiar", action="store_true", help="Vaciar train/ y val/ antes de empezar")
    return p.parse_args()


def main() -> None:
    args = argumentos()
    random.seed(args.semilla)
    detector = DetectorRostros() if args.recortar else None
    if detector:
        print(f"Detector de rostros: {detector.tipo}")

    total = {"train": 0, "val": 0, "descartadas": 0, "sin_rostro": 0}
    for clase in CLASES:
        origen = args.crudo / clase
        if not origen.is_dir():
            raise SystemExit(f"Falta la carpeta {origen}")

        imagenes = sorted(p for p in origen.rglob("*") if p.suffix.lower() in EXTENSIONES)
        if not imagenes:
            raise SystemExit(f"No hay imágenes en {origen}")
        random.shuffle(imagenes)

        corte = max(1, int(len(imagenes) * args.val))
        reparto = {"val": imagenes[:corte], "train": imagenes[corte:]}

        for particion, archivos in reparto.items():
            destino = args.destino / particion / clase
            if args.limpiar and destino.exists():
                shutil.rmtree(destino)
            destino.mkdir(parents=True, exist_ok=True)

            for i, imagen in enumerate(archivos, 1):
                ok, uso_completa = _copiar(imagen, destino, detector, args.tam_min,
                                           args.sin_rostro)
                total[particion if ok else "descartadas"] += 1
                total["sin_rostro"] += uso_completa
                if i % 200 == 0:
                    print(f"  {clase}/{particion}: {i}/{len(archivos)}", end="\r", flush=True)
            print(f"  {clase}/{particion}: {len(archivos)} procesadas" + " " * 20)

    print(f"\nListo · train {total['train']} · val {total['val']} · "
          f"descartadas {total['descartadas']}")
    if total["sin_rostro"]:
        print(f"{total['sin_rostro']} imágenes sin rostro detectado se usaron enteras "
              f"(--sin-rostro descartar para excluirlas)")
    meta = args.crudo / "metadatos.csv"
    if meta.exists():
        shutil.copy2(meta, args.destino / "metadatos.csv")
        print(f"Metadatos (edad, etnia) copiados a {args.destino / 'metadatos.csv'}")

    print(f"Dataset en {args.destino}. Ahora: python entrenar.py --preset calidad")


def _copiar(origen: Path, destino: Path, detector: DetectorRostros | None, tam_min: int,
            sin_rostro: str) -> tuple[bool, bool]:
    """Devuelve (guardada, se_usó_la_imagen_completa)."""
    salida = destino / f"{origen.stem}{origen.suffix.lower()}"
    if detector is None:
        shutil.copy2(origen, salida)
        return True, False

    imagen = cv2.imread(str(origen))
    if imagen is None:
        return False, False

    cajas = detector.detectar(imagen)
    caja = _principal(cajas, imagen.shape[1], imagen.shape[0], tam_min)
    if caja is None:
        # Los datasets de rostros ya vienen recortados: descartarlos por no detectar
        # nada perdería muchas imágenes buenas (caras muy grandes, de perfil o borrosas).
        if sin_rostro == "descartar":
            return False, False
        recorte, completa = imagen, True
    else:
        recorte, completa = recortar_rostro(imagen, caja), False

    guardada = bool(cv2.imwrite(str(salida.with_suffix(".jpg")), recorte,
                                [cv2.IMWRITE_JPEG_QUALITY, 95]))
    return guardada, completa and guardada


def _principal(cajas: list, ancho: int, alto: int, tam_min: int):
    """Elige el rostro del sujeto etiquetado: grande, pero sobre todo centrado.

    En datasets como FairFace la foto está centrada en la persona etiquetada y puede
    haber transeúntes más grandes en un borde; quedarse con el mayor metería ruido.
    """
    candidatas = [c for c in cajas if c[2] >= tam_min and c[3] >= tam_min]
    if not candidatas:
        return None
    cx, cy = ancho / 2, alto / 2
    diagonal = (ancho**2 + alto**2) ** 0.5

    def puntuar(caja):
        x, y, w, h = caja
        distancia = (((x + w / 2) - cx) ** 2 + ((y + h / 2) - cy) ** 2) ** 0.5
        return (w * h) / (ancho * alto) / (1 + 2 * distancia / diagonal)

    return max(candidatas, key=puntuar)


if __name__ == "__main__":
    main()

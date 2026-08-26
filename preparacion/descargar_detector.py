"""Descarga el detector de rostros YuNet (~230 KB) en modelos/.

Sin este archivo el sistema funciona igual, pero con el detector Haar de OpenCV,
bastante más impreciso. Si no hay internet se puede bajar a mano desde:
https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
y dejarlo en modelos/face_detection_yunet_2023mar.onnx
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "modelos" / "face_detection_yunet_2023mar.onnx"
RUTA = "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

# El archivo está en Git LFS: raw.githubusercontent.com devuelve solo el puntero,
# así que hay que usar media.githubusercontent.com (o la redirección de github.com/raw).
URLS = [
    f"https://media.githubusercontent.com/media/opencv/opencv_zoo/main/{RUTA}",
    f"https://github.com/opencv/opencv_zoo/raw/main/{RUTA}",
]


def main() -> None:
    if DESTINO.exists():
        print(f"Ya existe: {DESTINO}")
        return

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        print(f"Descargando {url} ...")
        try:
            urllib.request.urlretrieve(url, DESTINO)
        except Exception as error:  # noqa: BLE001 - queremos el mensaje tal cual
            print(f"  falló: {error}")
            DESTINO.unlink(missing_ok=True)
            continue

        tam = DESTINO.stat().st_size
        if tam < 50_000:  # puntero de Git LFS o error HTML
            print("  el archivo no es válido (puntero de Git LFS)")
            DESTINO.unlink()
            continue

        print(f"Listo: {DESTINO} ({tam / 1024:.0f} KB)")
        return

    sys.exit(f"No se pudo descargar de ninguna réplica. Bájalo a mano y déjalo en {DESTINO}")


if __name__ == "__main__":
    main()

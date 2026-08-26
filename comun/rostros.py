"""Detección de rostros.

Usa YuNet (ONNX, mucho más preciso) si el modelo está descargado en `modelos/`,
y si no cae automáticamente al clasificador Haar que viene incluido con OpenCV.

    python entrenamiento/descargar_detector.py   # descarga YuNet (opcional pero recomendado)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
RUTA_YUNET = RAIZ / "modelos" / "face_detection_yunet_2023mar.onnx"

Caja = tuple[int, int, int, int]  # (x, y, ancho, alto)


class DetectorRostros:
    def __init__(self, umbral: float = 0.6, ruta_yunet: str | Path | None = None):
        ruta = Path(ruta_yunet) if ruta_yunet else RUTA_YUNET
        self.umbral = umbral
        if ruta.exists():
            self.tipo = "yunet"
            self._det = cv2.FaceDetectorYN.create(
                str(ruta), "", (320, 320), umbral, 0.3, 5000
            )
        else:
            self.tipo = "haar"
            cascada = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._det = cv2.CascadeClassifier(str(cascada))
            if self._det.empty():
                raise RuntimeError(f"No se pudo cargar la cascada Haar en {cascada}")

    def detectar(self, imagen_bgr: np.ndarray) -> list[Caja]:
        """Devuelve las cajas de los rostros encontrados, ordenadas de mayor a menor."""
        alto, ancho = imagen_bgr.shape[:2]
        if ancho == 0 or alto == 0:
            return []

        if self.tipo == "yunet":
            self._det.setInputSize((ancho, alto))
            _, caras = self._det.detect(imagen_bgr)
            cajas = []
            if caras is not None:
                for cara in caras:
                    x, y, w, h = cara[:4].astype(int)
                    cajas.append(_ajustar(x, y, w, h, ancho, alto))
        else:
            gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
            gris = cv2.equalizeHist(gris)
            detecciones = self._det.detectMultiScale(
                gris, scaleFactor=1.15, minNeighbors=6, minSize=(60, 60)
            )
            cajas = [_ajustar(*map(int, d), ancho, alto) for d in detecciones]

        cajas = [c for c in cajas if c[2] > 0 and c[3] > 0]
        cajas.sort(key=lambda c: c[2] * c[3], reverse=True)
        return cajas


def _ajustar(x: int, y: int, w: int, h: int, ancho: int, alto: int) -> Caja:
    x = max(0, min(x, ancho - 1))
    y = max(0, min(y, alto - 1))
    w = max(0, min(w, ancho - x))
    h = max(0, min(h, alto - y))
    return (x, y, w, h)


def recortar_rostro(imagen_bgr: np.ndarray, caja: Caja, margen: float = 0.25) -> np.ndarray:
    """Recorta el rostro añadiendo margen alrededor (pelo, mentón y cuello ayudan al modelo)."""
    alto, ancho = imagen_bgr.shape[:2]
    x, y, w, h = caja
    mx, my = int(w * margen), int(h * margen)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(ancho, x + w + mx), min(alto, y + h + my)
    return imagen_bgr[y1:y2, x1:x2]

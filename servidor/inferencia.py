"""Inferencia: detecta rostros y clasifica cada uno como hombre o mujer.

Carga el checkpoint .pt generado por el entrenamiento, o un .onnx exportado con
entrenamiento/exportar.py (en ese caso hace falta onnxruntime).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import cv2
import numpy as np

from comun.rostros import DetectorRostros, recortar_rostro


class Clasificador:
    def __init__(self, ruta_modelo: Path, dispositivo: str = "auto",
                 umbral_rostro: float = 0.6, margen: float = 0.25):
        self.ruta = Path(ruta_modelo)
        if self.ruta.is_dir():
            self.ruta = self.ruta / "mejor.pt"
        if not self.ruta.exists():
            raise FileNotFoundError(
                f"No existe el modelo {self.ruta}.\n"
                "Entrénalo primero:  python entrenar.py --config configs/genero.yaml"
            )
        self.margen = margen
        self.detector = DetectorRostros(umbral=umbral_rostro)
        self.backend = "onnx" if self.ruta.suffix == ".onnx" else "torch"

        if self.backend == "onnx":
            self._cargar_onnx()
        else:
            self._cargar_torch(dispositivo)

        self.media = np.array(self.meta["media"], dtype=np.float32).reshape(3, 1, 1)
        self.desv = np.array(self.meta["desv"], dtype=np.float32).reshape(3, 1, 1)

    # ------------------------------------------------------------------ carga

    def _cargar_torch(self, dispositivo: str) -> None:
        import torch

        from nucleo.carga import cargar_modelo, metadatos

        self.torch = torch
        if dispositivo == "auto":
            dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        self.dispositivo = dispositivo
        self.modelo, ckpt = cargar_modelo(self.ruta, dispositivo)
        self.meta = metadatos(ckpt)
        self.meta["acc_val"] = self.meta.get("metricas", {}).get("acc")

    def _cargar_onnx(self) -> None:
        import onnxruntime as ort

        meta_json = next((r for r in (self.ruta.with_name("modelo.meta.json"),
                                      self.ruta.with_suffix(".meta.json")) if r.exists()), None)
        if meta_json is None:
            raise FileNotFoundError(
                f"Falta modelo.meta.json junto a {self.ruta} (lo genera exportar.py)")
        self.meta = json.loads(meta_json.read_text())
        self.meta["acc_val"] = self.meta.get("metricas", {}).get("acc")
        proveedores = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                       if p in ort.get_available_providers()]
        self.sesion = ort.InferenceSession(str(self.ruta), providers=proveedores)
        self.dispositivo = "cuda" if proveedores[0].startswith("CUDA") else "cpu"

    # ------------------------------------------------------------- inferencia

    def predecir(self, imagen_bgr: np.ndarray, detectar: bool = True,
                 max_rostros: int = 5) -> dict:
        inicio = time.perf_counter()
        alto, ancho = imagen_bgr.shape[:2]

        if detectar:
            cajas = self.detector.detectar(imagen_bgr)[:max_rostros]
            recortes = [recortar_rostro(imagen_bgr, c, self.margen) for c in cajas]
        else:
            cajas = [(0, 0, ancho, alto)]
            recortes = [imagen_bgr]

        recortes = [r for r in recortes if r.size > 0]
        if not recortes:
            return {"rostros": [], "ms": (time.perf_counter() - inicio) * 1000,
                    "ancho": ancho, "alto": alto}

        lote = np.stack([self._preprocesar(r) for r in recortes])
        probabilidades = self._inferir(lote)

        clases = self.meta["clases"]
        rostros = []
        for caja, probs in zip(cajas, probabilidades):
            indice = int(np.argmax(probs))
            rostros.append({
                "caja": {"x": int(caja[0]), "y": int(caja[1]),
                         "ancho": int(caja[2]), "alto": int(caja[3])},
                "etiqueta": clases[indice],
                "confianza": float(probs[indice]),
                "probabilidades": {c: float(p) for c, p in zip(clases, probs)},
            })

        return {"rostros": rostros, "ms": (time.perf_counter() - inicio) * 1000,
                "ancho": ancho, "alto": alto}

    def _preprocesar(self, recorte_bgr: np.ndarray) -> np.ndarray:
        """Mismo preprocesado que la validación: redimensionar lado corto + recorte central."""
        tam = self.meta["tam_img"]
        alto, ancho = recorte_bgr.shape[:2]
        escala = int(tam * 1.14) / min(alto, ancho)
        nueva = (max(tam, int(round(ancho * escala))), max(tam, int(round(alto * escala))))
        interpolacion = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
        imagen = cv2.resize(recorte_bgr, nueva, interpolation=interpolacion)

        x = (imagen.shape[1] - tam) // 2
        y = (imagen.shape[0] - tam) // 2
        imagen = imagen[y:y + tam, x:x + tam]

        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return (rgb.transpose(2, 0, 1) - self.media) / self.desv

    def _inferir(self, lote: np.ndarray) -> np.ndarray:
        # La temperatura viene del calibrado sobre validación: sin ella los porcentajes
        # que ve el usuario son sistemáticamente más altos que el acierto real.
        temperatura = float(self.meta.get("temperatura") or 1.0)
        if self.backend == "onnx":
            logits = self.sesion.run(None, {self.sesion.get_inputs()[0].name: lote})[0]
            return _softmax(logits / temperatura)
        with self.torch.no_grad():
            tensor = self.torch.from_numpy(lote).to(self.dispositivo)
            salida = self.modelo(tensor)
            if isinstance(salida, dict):
                salida = salida["principal"]
            return (salida / temperatura).softmax(1).cpu().numpy()

    def info(self) -> dict:
        return {
            "modelo": self.ruta.name,
            "backend": self.backend,
            "dispositivo": self.dispositivo,
            "arquitectura": self.meta["arquitectura"],
            "clases": self.meta["clases"],
            "tam_img": self.meta["tam_img"],
            "detector": self.detector.tipo,
            "entrenado": self.meta.get("fecha"),
            "acc_val": self.meta.get("acc_val"),
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def decodificar(datos: bytes) -> np.ndarray | None:
    buffer = np.frombuffer(datos, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

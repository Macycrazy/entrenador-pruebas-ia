"""Servicios de inferencia del panel: uno por tarea, con carga perezosa.

Todos comparten la misma forma: se construyen con su Vista, cargan el modelo la primera
vez que se usan y exponen `info()` para el chip de estado. Si el modelo no existe, el
servicio lanza FileNotFoundError y la API responde 503 con instrucciones.
"""

from __future__ import annotations

import io
import os
import sys
from collections import OrderedDict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np
import torch
from PIL import Image

from nucleo.carga import cargar_modelo, metadatos
from nucleo.config import Config


class Servicio:
    def __init__(self, vista):
        self.vista = vista
        self.modelo = None
        self.meta = {}
        self.cfg = None
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

    # --- ciclo de vida ---------------------------------------------------

    def asegurar(self):
        if self.modelo is None:
            self.cargar()
        return self

    def cargar(self) -> None:
        self.modelo, ckpt = cargar_modelo(self.vista.ruta_modelo, self.dispositivo)
        self.meta = metadatos(ckpt)
        self.cfg = Config(ckpt["config"])

    def info(self) -> dict:
        self.asegurar()
        return {
            "listo": True, "tarea": self.vista.tarea,
            "arquitectura": self.meta.get("arquitectura"),
            "clases": self.meta.get("clases"),
            "dispositivo": self.dispositivo,
            "entrenado": self.meta.get("fecha"),
            "acc_val": (self.meta.get("metricas") or {}).get("acc"),
            "extra": {},
        }

    def liberar(self) -> None:
        """Suelta el modelo para que el recolector y la GPU recuperen la memoria."""
        for atributo in ("modelo", "tuberia", "vocoder", "procesador", "lector",
                         "extractor", "vae", "codificador", "tokenizador"):
            if getattr(self, atributo, None) is not None:
                setattr(self, atributo, None)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- utilidades ------------------------------------------------------

    def _transformacion(self, tam=None):
        from tareas.imagen_clasificacion import aumentos
        return aumentos.validacion(self.cfg, tam or self.meta.get("tam_img", 224))

    @staticmethod
    def abrir(datos: bytes) -> Image.Image:
        return Image.open(io.BytesIO(datos)).convert("RGB")


# ===========================================================================
# Clasificación de imagen sobre rostros: género, atributos, detección de vida
# ===========================================================================

class ServicioImagen(Servicio):
    """Detecta rostros y clasifica cada uno. Con modelos multitarea añade edad y etnia."""

    def __init__(self, vista, recortar_rostro: bool = True):
        super().__init__(vista)
        self.recortar = recortar_rostro
        self.detector = None

    def cargar(self) -> None:
        super().cargar()
        if self.recortar:
            from comun.rostros import DetectorRostros
            self.detector = DetectorRostros()
        self.transformacion = self._transformacion()
        self.extras = self.meta.get("extras") or {}

    def info(self) -> dict:
        datos = super().info()
        datos["detector"] = self.detector.tipo if self.detector else "ninguno"
        datos["extra"] = {k: list(v) for k, v in self.extras.items()}
        return datos

    @torch.no_grad()
    def predecir(self, imagen_bytes: bytes, detectar: bool = True, max_rostros: int = 5) -> dict:
        self.asegurar()
        imagen = self.abrir(imagen_bytes)
        ancho, alto = imagen.size

        cajas, recortes = self._recortes(imagen, detectar, max_rostros)
        if not recortes:
            return {"rostros": [], "ancho": ancho, "alto": alto}

        lote = torch.stack([self.transformacion(r) for r in recortes]).to(self.dispositivo)
        salida = self.modelo(lote, None, solo_logits=False) if self.extras else self.modelo(lote)
        logits = salida["principal"] if isinstance(salida, dict) else salida

        temperatura = float(self.meta.get("temperatura") or 1.0)
        probs = (logits / temperatura).softmax(1).cpu().numpy()
        clases = self.meta["clases"]

        extras_probs = {}
        if isinstance(salida, dict):
            for nombre in self.extras:
                if nombre in salida:
                    extras_probs[nombre] = salida[nombre].softmax(1).cpu().numpy()

        rostros = []
        for i, caja in enumerate(cajas):
            indice = int(probs[i].argmax())
            registro = {
                # int() explícito: el detector devuelve int64 de numpy y json no lo serializa
                "caja": _caja(caja),
                "etiqueta": clases[indice], "confianza": float(probs[i][indice]),
                "probabilidades": {c: float(p) for c, p in zip(clases, probs[i])},
                "extra": {},
            }
            for nombre, matriz in extras_probs.items():
                valores = self.extras[nombre]
                j = int(matriz[i].argmax())
                registro["extra"][nombre] = {
                    "etiqueta": valores[j] if j < len(valores) else str(j),
                    "confianza": float(matriz[i][j]),
                }
            rostros.append(registro)
        return {"rostros": rostros, "ancho": ancho, "alto": alto}

    def _recortes(self, imagen: Image.Image, detectar: bool, max_rostros: int):
        if not (detectar and self.detector):
            return [(0, 0, *imagen.size)], [imagen]

        import cv2
        from comun.rostros import recortar_rostro
        bgr = cv2.cvtColor(np.asarray(imagen), cv2.COLOR_RGB2BGR)
        cajas = self.detector.detectar(bgr)[:max_rostros]
        recortes = [Image.fromarray(cv2.cvtColor(recortar_rostro(bgr, c), cv2.COLOR_BGR2RGB))
                    for c in cajas]
        pares = [(c, r) for c, r in zip(cajas, recortes) if r.width and r.height]
        return [p[0] for p in pares], [p[1] for p in pares]


# ===========================================================================
# Reconocimiento facial
# ===========================================================================

class ServicioRostros(Servicio):
    """Identificación 1:N contra la galería y verificación 1:1 entre dos fotos."""

    def __init__(self, vista, galeria: Path | None = None):
        super().__init__(vista)
        self.ruta_galeria = galeria or RAIZ / "modelos" / "galeria.npz"
        self.galeria = None
        self.detector = None

    def cargar(self) -> None:
        from comun.rostros import DetectorRostros
        from galeria_rostros import Extractor

        # El extractor sabe cargar tanto un checkpoint como un backbone suelto
        # (rostro:facenet), que es lo que permite usar la vista sin entrenar nada.
        origen = (str(self.vista.ruta_modelo) if self.vista.entrenada()
                  else "rostro:facenet")
        self.extractor = Extractor(origen)
        self.modelo = self.extractor.modelo
        self.meta = self.extractor.meta
        self.cfg = self.extractor.cfg
        self.detector = DetectorRostros()
        self.cargar_galeria()

    def cargar_galeria(self) -> None:
        if not self.ruta_galeria.exists():
            self.galeria = None
            return
        datos = np.load(self.ruta_galeria, allow_pickle=False)
        self.galeria = {"nombres": [str(n) for n in datos["nombres"]],
                        "vectores": datos["vectores"],
                        "umbral": float(datos["umbral"])}

    def info(self) -> dict:
        self.asegurar()
        return {
            "listo": True, "tarea": self.vista.tarea,
            "arquitectura": self.meta.get("arquitectura"),
            "dispositivo": self.dispositivo,
            "entrenado": self.meta.get("fecha"),
            "acc_val": (self.meta.get("metricas") or {}).get("rank1"),
            "detector": self.detector.tipo,
            "extra": {
                "inscritos": len(self.galeria["nombres"]) if self.galeria else 0,
                "umbral": self.galeria["umbral"] if self.galeria else
                          self.cfg.rostros.umbral_similitud,
                "galeria": str(self.ruta_galeria),
                "sin_entrenar": not self.vista.entrenada(),
            },
        }

    @torch.no_grad()
    def vector(self, imagen_bytes: bytes):
        """Devuelve (embedding, caja del rostro, tamaño de la imagen)."""
        self.asegurar()
        import cv2
        from comun.rostros import recortar_rostro

        imagen = self.abrir(imagen_bytes)
        bgr = cv2.cvtColor(np.asarray(imagen), cv2.COLOR_RGB2BGR)
        cajas = self.detector.detectar(bgr)
        if cajas:
            recorte = Image.fromarray(
                cv2.cvtColor(recortar_rostro(bgr, cajas[0]), cv2.COLOR_BGR2RGB))
            caja = cajas[0]
        else:
            recorte, caja = imagen, (0, 0, *imagen.size)

        x = self.extractor.transformacion(recorte).unsqueeze(0).to(self.dispositivo)
        base = getattr(self.modelo, "_orig_mod", self.modelo)
        emb = base.embeddings(x) if hasattr(base, "embeddings") else \
            torch.nn.functional.normalize(self.modelo(x, None, solo_logits=False)["rasgos"])
        return emb.squeeze(0).float().cpu().numpy(), caja, imagen.size

    def identificar(self, imagen_bytes: bytes, umbral: float | None = None,
                    top: int = 4) -> dict:
        vector, caja, (ancho, alto) = self.vector(imagen_bytes)
        if self.galeria is None:
            return {"error": "galeria_vacia", "caja": _caja(caja),
                    "ancho": ancho, "alto": alto}
        umbral = umbral if umbral is not None else self.galeria["umbral"]
        similitudes = self.galeria["vectores"] @ vector
        orden = np.argsort(-similitudes)[:top]
        mejor = int(orden[0])
        return {
            "caja": _caja(caja), "ancho": ancho, "alto": alto,
            "identidad": self.galeria["nombres"][mejor] if similitudes[mejor] >= umbral else None,
            "similitud": float(similitudes[mejor]),
            "umbral": float(umbral),
            "candidatos": [{"nombre": self.galeria["nombres"][int(i)],
                            "similitud": float(similitudes[int(i)])} for i in orden],
        }

    def verificar(self, a: bytes, b: bytes, umbral: float | None = None) -> dict:
        va = self.vector(a)[0]
        vb = self.vector(b)[0]
        umbral = umbral if umbral is not None else (
            self.galeria["umbral"] if self.galeria else self.cfg.rostros.umbral_similitud)
        similitud = float(va @ vb)
        return {"similitud": similitud, "umbral": float(umbral),
                "misma_persona": similitud >= umbral}

    def inscribir(self, nombre: str, imagenes: list[bytes]) -> dict:
        vectores = [self.vector(datos)[0] for datos in imagenes]
        if not vectores:
            raise ValueError("No se pudo extraer ningún rostro")
        medio = np.mean(vectores, axis=0)
        medio /= np.linalg.norm(medio) + 1e-9

        if self.galeria is None:
            self.galeria = {"nombres": [], "vectores": np.zeros((0, len(medio))),
                            "umbral": float(self.cfg.rostros.umbral_similitud)}
        if nombre in self.galeria["nombres"]:        # actualizar en vez de duplicar
            indice = self.galeria["nombres"].index(nombre)
            self.galeria["vectores"][indice] = medio
        else:
            self.galeria["nombres"].append(nombre)
            self.galeria["vectores"] = np.vstack([self.galeria["vectores"], medio])

        self.ruta_galeria.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.ruta_galeria, nombres=np.array(self.galeria["nombres"]),
                 vectores=self.galeria["vectores"], conteos=np.zeros(len(self.galeria["nombres"])),
                 modelo=str(self.vista.ruta_modelo), umbral=self.galeria["umbral"])
        return {"nombre": nombre, "fotos": len(vectores),
                "inscritos": len(self.galeria["nombres"])}

    def olvidar(self, nombre: str) -> dict:
        if not self.galeria or nombre not in self.galeria["nombres"]:
            raise ValueError(f"{nombre} no está en la galería")
        indice = self.galeria["nombres"].index(nombre)
        self.galeria["nombres"].pop(indice)
        self.galeria["vectores"] = np.delete(self.galeria["vectores"], indice, axis=0)
        np.savez(self.ruta_galeria, nombres=np.array(self.galeria["nombres"]),
                 vectores=self.galeria["vectores"], conteos=np.zeros(len(self.galeria["nombres"])),
                 modelo=str(self.vista.ruta_modelo), umbral=self.galeria["umbral"])
        return {"inscritos": len(self.galeria["nombres"])}


def _caja(caja) -> dict:
    x, y, w, h = caja
    return {"x": int(x), "y": int(y), "ancho": int(w), "alto": int(h)}


# ===========================================================================
# Detección y segmentación
# ===========================================================================

class ServicioDeteccion(Servicio):
    def cargar(self) -> None:
        import torch as _t
        from tareas.vision_deteccion.modelos import crear_modelo

        ckpt = _t.load(self.vista.ruta_modelo, map_location=self.dispositivo,
                       weights_only=False)
        self.cfg = Config(ckpt["config"])
        self.clases = ckpt.get("clases_deteccion") or ckpt.get("clases", [])
        self.modelo = crear_modelo(self.cfg, len(self.clases) + 1)
        self.modelo.load_state_dict(ckpt["state_dict"])
        self.modelo.eval().to(self.dispositivo)
        self.meta = {"arquitectura": ckpt.get("arquitectura"), "clases": self.clases,
                     "fecha": ckpt.get("fecha"), "metricas": ckpt.get("metricas", {})}

    def info(self) -> dict:
        datos = super().info()
        datos["acc_val"] = (self.meta.get("metricas") or {}).get("map50")
        return datos

    @torch.no_grad()
    def predecir(self, imagen_bytes: bytes, umbral: float = 0.5) -> dict:
        self.asegurar()
        import torchvision.transforms.functional as TF
        imagen = self.abrir(imagen_bytes)
        x = TF.to_tensor(imagen).to(self.dispositivo)
        prediccion = self.modelo([x])[0]

        objetos = []
        for caja, etiqueta, score in zip(prediccion["boxes"].cpu().numpy(),
                                         prediccion["labels"].cpu().numpy(),
                                         prediccion["scores"].cpu().numpy()):
            if score < umbral:
                continue
            indice = int(etiqueta) - 1
            objetos.append({
                "caja": {"x": float(caja[0]), "y": float(caja[1]),
                         "ancho": float(caja[2] - caja[0]), "alto": float(caja[3] - caja[1])},
                "clase": self.clases[indice] if 0 <= indice < len(self.clases) else str(etiqueta),
                "indice": indice, "confianza": float(score),
            })
        return {"objetos": objetos, "ancho": imagen.width, "alto": imagen.height}


class ServicioSegmentacion(Servicio):
    def cargar(self) -> None:
        from tareas.vision_segmentacion import _cambiar_cabezas
        from torchvision.models import segmentation

        ckpt = torch.load(self.vista.ruta_modelo, map_location=self.dispositivo,
                          weights_only=False)
        self.cfg = Config(ckpt["config"])
        self.clases = ckpt.get("clases_segmentacion") or ckpt.get("clases", [])
        nombre = ckpt.get("arquitectura", self.cfg.modelo.arquitectura)
        aux = "lraspp" not in nombre
        self.modelo = getattr(segmentation, nombre)(
            weights=None, **({"aux_loss": True} if aux else {}))
        _cambiar_cabezas(self.modelo, len(self.clases))
        self.modelo.load_state_dict(ckpt["state_dict"])
        self.modelo.eval().to(self.dispositivo)
        self.meta = {"arquitectura": nombre, "clases": self.clases,
                     "fecha": ckpt.get("fecha"), "metricas": ckpt.get("metricas", {}),
                     "tam_img": ckpt.get("tam_img", self.cfg.datos.tam_img)}

    def info(self) -> dict:
        datos = super().info()
        datos["acc_val"] = (self.meta.get("metricas") or {}).get("miou")
        return datos

    @torch.no_grad()
    def predecir(self, imagen_bytes: bytes) -> dict:
        """Devuelve la máscara como PNG en color y el reparto de píxeles por clase."""
        self.asegurar()
        import torchvision.transforms.functional as TF
        from tareas.imagen_clasificacion.aumentos import DESV, MEDIA

        imagen = self.abrir(imagen_bytes)
        tam = self.meta["tam_img"]
        pequena = TF.resize(imagen, [tam, tam])
        x = TF.normalize(TF.to_tensor(pequena), MEDIA, DESV).unsqueeze(0).to(self.dispositivo)
        prediccion = self.modelo(x)["out"].argmax(1)[0].cpu().numpy().astype(np.uint8)

        paleta = _paleta(len(self.clases))
        color = paleta[prediccion]
        mascara = Image.fromarray(color).resize(imagen.size, Image.NEAREST)
        buffer = io.BytesIO()
        mascara.save(buffer, format="PNG")

        total = prediccion.size
        reparto = {self.clases[i]: float((prediccion == i).sum() / total)
                   for i in range(len(self.clases)) if (prediccion == i).any()}
        return {"png": buffer.getvalue(), "reparto": reparto,
                "colores": {c: "#%02x%02x%02x" % tuple(paleta[i])
                            for i, c in enumerate(self.clases)},
                "ancho": imagen.width, "alto": imagen.height}


def _paleta(n: int) -> np.ndarray:
    base = np.array([[0, 0, 0], [56, 189, 248], [167, 139, 250], [251, 191, 36],
                     [52, 211, 153], [248, 113, 113], [244, 114, 182], [96, 165, 250]],
                    dtype=np.uint8)
    if n <= len(base):
        return base[:n]
    extra = np.random.default_rng(0).integers(60, 255, (n - len(base), 3), dtype=np.uint8)
    return np.vstack([base, extra])


# ===========================================================================
# Super-resolución
# ===========================================================================

class ServicioSuperResolucion(Servicio):
    def cargar(self) -> None:
        from tareas.imagen_superresolucion.modelos import SuperResolucion

        ckpt = torch.load(self.vista.ruta_modelo, map_location=self.dispositivo,
                          weights_only=False)
        self.cfg = Config(ckpt["config"])
        self.escala = ckpt.get("escala", self.cfg.superresolucion.escala)
        self.modelo = SuperResolucion(self.escala, ckpt.get("canales", 64),
                                      ckpt.get("bloques", 8))
        self.modelo.load_state_dict(ckpt["state_dict"])
        self.modelo.eval().to(self.dispositivo)
        self.meta = {"arquitectura": ckpt.get("arquitectura"), "clases": [],
                     "fecha": ckpt.get("fecha"), "metricas": ckpt.get("metricas", {})}

    def info(self) -> dict:
        datos = super().info()
        metricas = self.meta.get("metricas") or {}
        datos["acc_val"] = None
        datos["extra"] = {"escala": self.escala,
                          "psnr": round(metricas.get("psnr", 0), 2),
                          "mejora_db": round(metricas.get("mejora_db", 0), 2)}
        return datos

    @torch.no_grad()
    def predecir(self, imagen_bytes: bytes, lado_max: int = 512) -> dict:
        """Devuelve la imagen agrandada por el modelo y, para comparar, la bicúbica."""
        import time

        import torchvision.transforms.functional as TF
        self.asegurar()
        imagen = self.abrir(imagen_bytes)
        # Se limita la entrada: agrandar x4 una foto enorme se come la memoria
        if max(imagen.size) > lado_max:
            imagen = TF.resize(imagen, [int(imagen.height * lado_max / max(imagen.size)),
                                        int(imagen.width * lado_max / max(imagen.size))])

        x = TF.to_tensor(imagen).unsqueeze(0).to(self.dispositivo)
        inicio = time.perf_counter()
        salida = self.modelo(x)[0].clamp(0, 1).cpu()
        ms = (time.perf_counter() - inicio) * 1000

        base = torch.nn.functional.interpolate(
            x, scale_factor=self.escala, mode="bicubic", align_corners=False)[0].clamp(0, 1).cpu()

        return {"mejorada": _png(TF.to_pil_image(salida)),
                "bicubica": _png(TF.to_pil_image(base)),
                "original": _png(imagen),
                "escala": self.escala, "ms": ms,
                "entrada": list(imagen.size), "salida": [salida.shape[2], salida.shape[1]]}


def _png(imagen: Image.Image) -> bytes:
    buffer = io.BytesIO()
    imagen.save(buffer, format="PNG")
    return buffer.getvalue()


# ===========================================================================
# Audio
# ===========================================================================

class ServicioAudio(Servicio):
    def cargar(self) -> None:
        super().cargar()
        from tareas.audio_clasificacion.datos import DatasetAudio
        self.constructor = DatasetAudio([], self.cfg, entrenando=False)

    @torch.no_grad()
    def predecir(self, audio_bytes: bytes, nombre: str = "audio.wav") -> dict:
        self.asegurar()
        import tempfile

        from tareas.audio_clasificacion.carga import cargar_audio

        sufijo = Path(nombre).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=sufijo, delete=False) as tmp:
            tmp.write(audio_bytes)
            ruta = Path(tmp.name)
        try:
            onda, sr = cargar_audio(ruta)
            if sr != self.cfg.audio.sr:
                onda = self.constructor._remuestrear(onda, sr, self.cfg.audio.sr)
            duracion = onda.shape[-1] / self.cfg.audio.sr
            onda = self.constructor._ajustar_duracion(onda)
            espectro = self.constructor.a_db(self.constructor.mel(onda))
            espectro = (espectro - espectro.mean()) / (espectro.std() + 1e-5)
            x = espectro.repeat(3, 1, 1).unsqueeze(0).to(self.dispositivo)

            logits = self.modelo(x)
            if isinstance(logits, dict):
                logits = logits["principal"]
            temperatura = float(self.meta.get("temperatura") or 1.0)
            probs = (logits / temperatura).softmax(1)[0].cpu().numpy()
        finally:
            ruta.unlink(missing_ok=True)

        clases = self.meta["clases"]
        indice = int(probs.argmax())
        return {"etiqueta": clases[indice], "confianza": float(probs[indice]),
                "probabilidades": {c: float(p) for c, p in zip(clases, probs)},
                "duracion": round(duracion, 2)}


# ===========================================================================
# Percepción sin entrenamiento: pose y profundidad
# ===========================================================================

# Esqueleto COCO: qué punto se une con cuál
HUESOS = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
          (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5),
          (4, 6)]
ARTICULACIONES = ["nariz", "ojo izq", "ojo der", "oreja izq", "oreja der",
                  "hombro izq", "hombro der", "codo izq", "codo der", "muñeca izq",
                  "muñeca der", "cadera izq", "cadera der", "rodilla izq", "rodilla der",
                  "tobillo izq", "tobillo der"]


class ServicioPose(Servicio):
    """Esqueleto de cada persona. Usa un modelo ya entrenado con COCO: no hay que
    entrenar nada, y sirve para postura, ergonomía o detectar caídas."""

    def cargar(self) -> None:
        from torchvision.models.detection import (KeypointRCNN_ResNet50_FPN_Weights,
                                                  keypointrcnn_resnet50_fpn)
        pesos = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self.modelo = keypointrcnn_resnet50_fpn(weights=pesos).eval().to(self.dispositivo)
        self.meta = {"arquitectura": "keypointrcnn_resnet50_fpn", "clases": ARTICULACIONES,
                     "fecha": "preentrenado (COCO)", "metricas": {}}

    def info(self) -> dict:
        datos = super().info()
        datos["acc_val"] = None
        datos["extra"] = {"sin_entrenar": True, "articulaciones": len(ARTICULACIONES),
                          "huesos": HUESOS, "nombres": ARTICULACIONES}
        return datos

    @torch.no_grad()
    def predecir(self, imagen_bytes: bytes, umbral: float = 0.8) -> dict:
        import torchvision.transforms.functional as TF
        self.asegurar()
        imagen = self.abrir(imagen_bytes)
        x = TF.to_tensor(imagen).to(self.dispositivo)
        salida = self.modelo([x])[0]

        personas = []
        for puntos, puntuacion, caja in zip(salida["keypoints"].cpu().numpy(),
                                            salida["scores"].cpu().numpy(),
                                            salida["boxes"].cpu().numpy()):
            if puntuacion < umbral:
                continue
            personas.append({
                "confianza": float(puntuacion),
                "caja": {"x": float(caja[0]), "y": float(caja[1]),
                         "ancho": float(caja[2] - caja[0]), "alto": float(caja[3] - caja[1])},
                "puntos": [{"x": float(p[0]), "y": float(p[1]), "visible": bool(p[2] > 0)}
                           for p in puntos],
            })
        return {"personas": personas, "ancho": imagen.width, "alto": imagen.height,
                "huesos": HUESOS, "nombres": ARTICULACIONES}


class ServicioProfundidad(Servicio):
    """Estima a qué distancia está cada píxel a partir de una sola foto."""

    MODELO = "depth-anything/Depth-Anything-V2-Small-hf"

    def cargar(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.procesador = AutoImageProcessor.from_pretrained(self.MODELO)
        self.modelo = AutoModelForDepthEstimation.from_pretrained(self.MODELO)
        self.modelo.eval().to(self.dispositivo)
        self.meta = {"arquitectura": self.MODELO, "clases": [],
                     "fecha": "preentrenado", "metricas": {}}

    def info(self) -> dict:
        datos = super().info()
        datos["acc_val"] = None
        datos["extra"] = {"sin_entrenar": True}
        return datos

    @torch.no_grad()
    def predecir(self, imagen_bytes: bytes) -> dict:
        import time
        self.asegurar()
        imagen = self.abrir(imagen_bytes)
        entradas = self.procesador(images=imagen, return_tensors="pt").to(self.dispositivo)

        inicio = time.perf_counter()
        profundidad = self.modelo(**entradas).predicted_depth
        ms = (time.perf_counter() - inicio) * 1000

        mapa = torch.nn.functional.interpolate(
            profundidad.unsqueeze(1), size=(imagen.height, imagen.width),
            mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
        # Se normaliza a 0-255 y se colorea: azul lejos, amarillo cerca
        normal = (mapa - mapa.min()) / (mapa.max() - mapa.min() + 1e-8)
        color = _colorear(normal)
        return {"png": _png(Image.fromarray(color)),
                "gris": _png(Image.fromarray((normal * 255).astype(np.uint8))),
                "ms": ms, "ancho": imagen.width, "alto": imagen.height}


def _colorear(normal: np.ndarray) -> np.ndarray:
    """Rampa de color sencilla (tipo inferno) sin depender de matplotlib."""
    paradas = np.array([[10, 12, 40], [60, 20, 110], [160, 40, 100],
                        [230, 90, 40], [250, 200, 60], [255, 255, 220]], dtype=np.float32)
    posicion = normal * (len(paradas) - 1)
    bajo = np.clip(posicion.astype(int), 0, len(paradas) - 2)
    peso = (posicion - bajo)[..., None]
    return (paradas[bajo] * (1 - peso) + paradas[bajo + 1] * peso).astype(np.uint8)


def _hay_avx2() -> bool:
    """Los kernels cuantizados de PyTorch necesitan AVX2; sin él, revientan."""
    try:
        with open("/proc/cpuinfo") as f:
            for linea in f:
                if linea.startswith("flags"):
                    return " avx2 " in f" {linea} "
    except OSError:
        pass
    return True     # fuera de Linux se asume que sí y se deja fallar de forma normal


class ServicioOCR(Servicio):
    """Lee el texto de una foto: cédulas, carnets, formularios, placas.

    Usa EasyOCR, que hace las dos partes del trabajo —encontrar dónde hay texto y leerlo—
    y trae modelos para español. Además busca patrones típicos (cédula, fechas, códigos)
    para no tener que rebuscar a mano en el resultado.
    """

    IDIOMAS = ["es", "en"]

    def cargar(self) -> None:
        try:
            import easyocr
        except ImportError:
            raise SystemExit("El OCR necesita:  pip install easyocr") from None
        # La cuantización acelera el reconocedor pero usa instrucciones AVX2. En un
        # procesador sin ellas el proceso muere con «instrucción ilegal» (SIGILL), sin
        # excepción ni rastro, así que se comprueba antes.
        self.lector = easyocr.Reader(
            self.IDIOMAS, gpu=self.dispositivo == "cuda", quantize=_hay_avx2(),
            model_storage_directory=str(RAIZ / "modelos" / "ocr"), verbose=False)
        self.modelo = self.lector
        self.meta = {"arquitectura": "easyocr (CRAFT + CRNN)", "clases": self.IDIOMAS,
                     "fecha": "preentrenado", "metricas": {}}

    def info(self) -> dict:
        datos = super().info()
        datos["acc_val"] = None
        datos["extra"] = {"sin_entrenar": True, "idiomas": self.IDIOMAS}
        return datos

    def predecir(self, imagen_bytes: bytes, umbral: float = 0.35) -> dict:
        import time
        self.asegurar()
        imagen = self.abrir(imagen_bytes)

        # Las fotos enormes disparan la memoria sin mejorar la lectura
        if max(imagen.size) > 1600:
            escala = 1600 / max(imagen.size)
            imagen = imagen.resize((int(imagen.width * escala), int(imagen.height * escala)))

        inicio = time.perf_counter()
        # workers=0 es importante: por defecto easyocr abre procesos de carga en paralelo
        # que multiplican la memoria y, en una máquina sin swap, el sistema mata el panel.
        crudo = self.lector.readtext(np.asarray(imagen), canvas_size=1280,
                                     workers=0, batch_size=1)
        ms = (time.perf_counter() - inicio) * 1000

        bloques = []
        for puntos, texto, confianza in crudo:
            if confianza < umbral or not texto.strip():
                continue
            xs = [float(p[0]) for p in puntos]
            ys = [float(p[1]) for p in puntos]
            bloques.append({
                "texto": texto.strip(), "confianza": float(confianza),
                "caja": {"x": min(xs), "y": min(ys),
                         "ancho": max(xs) - min(xs), "alto": max(ys) - min(ys)},
            })

        completo = "\n".join(b["texto"] for b in bloques)
        return {"bloques": bloques, "texto": completo, "campos": _extraer_campos(completo),
                "ms": ms, "ancho": imagen.width, "alto": imagen.height}


def _extraer_campos(texto: str) -> dict:
    """Busca los datos que suelen interesar de un documento venezolano."""
    import re

    campos = {}
    cedula = re.search(r"\b([VEJGvejg])[\s\-.]?(\d{1,2}[.\s]?\d{3}[.\s]?\d{3})\b", texto)
    if cedula:
        campos["cedula"] = f"{cedula.group(1).upper()}-{re.sub(r'[.\s]', '', cedula.group(2))}"
    else:
        suelto = re.search(r"\b(\d{1,2}[.]\d{3}[.]\d{3})\b", texto)
        if suelto:
            campos["cedula"] = suelto.group(1).replace(".", "")

    fecha = re.search(r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b", texto)
    if fecha:
        campos["fecha"] = fecha.group(1)
    correo = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", texto)
    if correo:
        campos["correo"] = correo.group(0)
    telefono = re.search(r"\b(0?4\d{2}[\s\-.]?\d{3}[\s\-.]?\d{4})\b", texto)
    if telefono:
        campos["telefono"] = re.sub(r"[\s\-.]", "", telefono.group(1))
    return campos


def _tensor(salida):
    """Según la versión de transformers, get_*_features devuelve el tensor o un objeto."""
    if isinstance(salida, torch.Tensor):
        return salida
    for atributo in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        valor = getattr(salida, atributo, None)
        if isinstance(valor, torch.Tensor):
            return valor if valor.dim() == 2 else valor.mean(1)
    raise TypeError(f"No sé extraer el vector de {type(salida).__name__}")


class ServicioImagenesTexto(Servicio):
    """Busca fotos describiéndolas con palabras, sin etiquetas ni carpetas.

    CLIP coloca imágenes y frases en el mismo espacio: la foto de una persona con gafas
    y el texto «a person wearing glasses» caen cerca. Primero se indexa una carpeta
    (una pasada por todas las fotos) y después buscar es instantáneo.
    """

    def __init__(self, vista):
        super().__init__(vista)
        self.indice_ruta = RAIZ / "modelos" / "indice_imagenes.npz"
        self.rutas: list[str] = []
        self.vectores = None

    def cargar(self) -> None:
        import copy

        from transformers import CLIPModel, CLIPProcessor

        from nucleo.config import Config as _Config, DEFECTOS

        self.cfg = _Config(copy.deepcopy(DEFECTOS))
        base = self.cfg.imagenes.modelo_base
        self.procesador = CLIPProcessor.from_pretrained(base)
        self.modelo = CLIPModel.from_pretrained(base).eval().to(self.dispositivo)
        self.meta = {"arquitectura": base, "clases": [], "fecha": "preentrenado",
                     "metricas": {}}
        self._cargar_indice()

    def _cargar_indice(self) -> None:
        if not self.indice_ruta.exists():
            self.rutas, self.vectores = [], None
            return
        datos = np.load(self.indice_ruta, allow_pickle=False)
        self.vectores = datos["vectores"]
        self.rutas = [str(r) for r in datos["rutas"]]

    def info(self) -> dict:
        datos = super().info()
        datos["acc_val"] = None
        datos["extra"] = {"sin_entrenar": True, "indexadas": len(self.rutas),
                          "carpetas": sorted({str(Path(r).parent) for r in self.rutas})[:8]}
        return datos

    @torch.no_grad()
    def indexar(self, carpeta: str, limite: int = 500) -> dict:
        import time
        self.asegurar()
        raiz = (RAIZ / carpeta).resolve()
        if not str(raiz).startswith(str(RAIZ)) or not raiz.is_dir():
            raise ValueError(f"Carpeta no válida: {carpeta}")

        extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        archivos = [p for p in sorted(raiz.rglob("*")) if p.suffix.lower() in extensiones]
        if not archivos:
            raise ValueError(f"No hay imágenes en {carpeta}")
        archivos = archivos[:limite]

        inicio = time.perf_counter()
        vectores = []
        for comienzo in range(0, len(archivos), self.cfg.imagenes.lote):
            lote = archivos[comienzo:comienzo + self.cfg.imagenes.lote]
            imagenes = []
            for ruta in lote:
                with Image.open(ruta) as bruta:
                    imagenes.append(bruta.convert("RGB"))
            entradas = self.procesador(images=imagenes, return_tensors="pt").to(self.dispositivo)
            rasgos = _tensor(self.modelo.get_image_features(**entradas))
            vectores.append(torch.nn.functional.normalize(rasgos, dim=1).cpu().numpy())

        nuevos = np.vstack(vectores)
        rutas = [str(p.relative_to(RAIZ)) for p in archivos]
        # Se reemplaza lo que hubiera de esa carpeta en vez de acumular duplicados
        conservar = [i for i, r in enumerate(self.rutas) if not r.startswith(carpeta)]
        anteriores = self.vectores[conservar] if (self.vectores is not None and conservar) else None
        self.rutas = [self.rutas[i] for i in conservar] + rutas
        self.vectores = nuevos if anteriores is None else np.vstack([anteriores, nuevos])

        self.indice_ruta.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.indice_ruta, vectores=self.vectores, rutas=np.array(self.rutas))
        return {"carpeta": carpeta, "indexadas": len(rutas), "total": len(self.rutas),
                "segundos": round(time.perf_counter() - inicio, 1)}

    @torch.no_grad()
    def buscar(self, texto: str, cuantos: int | None = None) -> dict:
        import time
        self.asegurar()
        if self.vectores is None or not len(self.rutas):
            return {"resultados": [], "detalle": "todavía no hay imágenes indexadas"}

        inicio = time.perf_counter()
        entradas = self.procesador(text=[texto], return_tensors="pt", padding=True,
                                   truncation=True).to(self.dispositivo)
        rasgos = _tensor(self.modelo.get_text_features(**entradas))
        vector = torch.nn.functional.normalize(rasgos, dim=1)[0].cpu().numpy()

        similitudes = self.vectores @ vector
        orden = np.argsort(-similitudes)[:cuantos or self.cfg.imagenes.resultados]
        return {"resultados": [{"ruta": self.rutas[int(i)],
                                "similitud": float(similitudes[int(i)])} for i in orden],
                "ms": (time.perf_counter() - inicio) * 1000, "indexadas": len(self.rutas)}


class ServicioAnomalias(Servicio):
    """Dice si una imagen se sale de lo normal y señala dónde."""

    def cargar(self) -> None:
        from tareas.imagen_anomalias import Autocodificador

        ckpt = torch.load(self.vista.ruta_modelo, map_location=self.dispositivo,
                          weights_only=False)
        self.cfg = Config(ckpt["config"])
        self.tam = ckpt.get("tam_img", self.cfg.anomalias.tam_img)
        self.modelo = Autocodificador(ckpt.get("canales", 32), ckpt.get("cuello", 64), self.tam)
        self.modelo.load_state_dict(ckpt["state_dict"])
        self.modelo.eval().to(self.dispositivo)
        metricas = ckpt.get("metricas", {}) or {}
        self.umbral = float(metricas.get("umbral") or 0.02)
        self.meta = {"arquitectura": "autocodificador", "clases": ["normal", "anomala"],
                     "fecha": ckpt.get("fecha"), "metricas": metricas}

    def info(self) -> dict:
        datos = super().info()
        metricas = self.meta.get("metricas") or {}
        datos["acc_val"] = metricas.get("auc")
        datos["extra"] = {"umbral": round(self.umbral, 5),
                          "error_normal": round(metricas.get("error_normal", 0), 5),
                          "detectadas": metricas.get("detectadas"),
                          "falsas_alarmas": metricas.get("falsas_alarmas"),
                          "tam_img": self.tam}
        return datos

    @torch.no_grad()
    def predecir(self, imagen_bytes: bytes, umbral: float | None = None) -> dict:
        import time

        import torchvision.transforms.functional as TF
        self.asegurar()
        imagen = self.abrir(imagen_bytes)
        recortada = TF.center_crop(TF.resize(imagen, self.tam), [self.tam, self.tam])
        x = TF.to_tensor(recortada).unsqueeze(0).to(self.dispositivo)

        inicio = time.perf_counter()
        reconstruida = self.modelo(x)
        ms = (time.perf_counter() - inicio) * 1000

        diferencia = (reconstruida - x).abs()[0]
        error = float(diferencia.mean())
        # Mapa por píxel: dónde falla la reconstrucción es dónde está lo raro
        mapa = diferencia.mean(0).cpu().numpy()
        mapa = (mapa - mapa.min()) / (mapa.max() - mapa.min() + 1e-8)
        color = _colorear(mapa)

        limite = umbral if umbral is not None else self.umbral
        return {"error": error, "umbral": float(limite), "anomala": error > limite,
                "veces_umbral": round(error / max(1e-9, limite), 2), "ms": ms,
                "mapa": _png(Image.fromarray(color).resize(imagen.size, Image.BILINEAR)),
                "reconstruida": _png(TF.to_pil_image(reconstruida[0].clamp(0, 1))),
                "entrada": _png(recortada)}


# ===========================================================================
# Datos tabulares y series temporales
# ===========================================================================

class ServicioTabular(Servicio):
    """Predice la columna objetivo para un caso nuevo introducido a mano."""

    def cargar(self) -> None:
        import torch.nn as nn

        ckpt = torch.load(self.vista.ruta_modelo, map_location="cpu", weights_only=False)
        self.cfg = Config(ckpt["config"])
        self.info_prep = {k: ckpt[k] for k in
                          ("entradas", "objetivo", "clases", "es_clasificacion",
                           "numericas", "categoricas", "n_entradas") if k in ckpt}

        salidas = len(self.info_prep["clases"]) if self.info_prep["es_clasificacion"] else 1
        capas, dentro = [], self.info_prep["n_entradas"]
        for ancho in self.cfg.tabular.capas:
            capas += [nn.Linear(dentro, ancho), nn.BatchNorm1d(ancho), nn.ReLU(),
                      nn.Dropout(self.cfg.tabular.dropout)]
            dentro = ancho
        self.modelo = nn.Sequential(*capas, nn.Linear(dentro, salidas))
        self.modelo.load_state_dict(ckpt["state_dict"])
        self.modelo.eval().to(self.dispositivo)
        self.meta = {"arquitectura": "mlp_tabular", "clases": self.info_prep["clases"],
                     "fecha": ckpt.get("fecha"), "metricas": ckpt.get("metricas", {})}

    def info(self) -> dict:
        datos = super().info()
        metricas = self.meta.get("metricas") or {}
        datos["acc_val"] = metricas.get("acc_balanceada") or metricas.get("acc")
        campos = []
        for columna, _, _ in self.info_prep["numericas"]:
            campos.append({"nombre": columna, "tipo": "numero"})
        for columna, valores in self.info_prep["categoricas"].items():
            campos.append({"nombre": columna, "tipo": "opcion", "valores": valores})
        datos["extra"] = {"objetivo": self.info_prep["objetivo"], "campos": campos,
                          "es_clasificacion": self.info_prep["es_clasificacion"],
                          "r2": metricas.get("r2"), "mae": metricas.get("mae")}
        return datos

    @torch.no_grad()
    def predecir(self, fila: dict) -> dict:
        self.asegurar()
        vector = np.zeros(self.info_prep["n_entradas"], dtype=np.float32)
        posicion = 0
        for columna, media, desv in self.info_prep["numericas"]:
            try:
                vector[posicion] = (float(fila.get(columna, media)) - media) / (desv or 1)
            except (TypeError, ValueError):
                vector[posicion] = 0.0
            posicion += 1
        for columna, valores in self.info_prep["categoricas"].items():
            if fila.get(columna) in valores:
                vector[posicion + valores.index(fila[columna])] = 1.0
            posicion += len(valores)

        salida = self.modelo(torch.tensor(vector).unsqueeze(0).to(self.dispositivo))
        if self.info_prep["es_clasificacion"]:
            temperatura = float((self.meta.get("metricas") or {}).get("temperatura") or 1.0)
            probs = (salida / temperatura).softmax(1)[0].cpu().numpy()
            clases = self.info_prep["clases"]
            i = int(probs.argmax())
            return {"objetivo": self.info_prep["objetivo"], "etiqueta": clases[i],
                    "confianza": float(probs[i]),
                    "probabilidades": {c: float(p) for c, p in zip(clases, probs)}}
        return {"objetivo": self.info_prep["objetivo"],
                "valor": float(salida.squeeze().cpu())}


class ServicioSeries(Servicio):
    """Continúa la serie: dados los últimos valores, predice los siguientes."""

    def cargar(self) -> None:
        import torch.nn as nn

        ckpt = torch.load(self.vista.ruta_modelo, map_location="cpu", weights_only=False)
        self.cfg = Config(ckpt["config"])
        self.media = ckpt.get("media", 0.0)
        self.desv = ckpt.get("desv", 1.0)
        self.ventana = ckpt.get("ventana", self.cfg.series.ventana)
        self.horizonte = ckpt.get("horizonte", self.cfg.series.horizonte)

        capas, dentro = [], self.ventana
        for ancho in self.cfg.series.capas:
            capas += [nn.Linear(dentro, ancho), nn.ReLU(), nn.Dropout(0.1)]
            dentro = ancho
        self.modelo = nn.Sequential(*capas, nn.Linear(dentro, self.horizonte))
        self.modelo.load_state_dict(ckpt["state_dict"])
        self.modelo.eval().to(self.dispositivo)
        self.meta = {"arquitectura": "mlp_series", "clases": [],
                     "fecha": ckpt.get("fecha"), "metricas": ckpt.get("metricas", {}),
                     "columna": ckpt.get("columna", "")}

    def info(self) -> dict:
        datos = super().info()
        metricas = self.meta.get("metricas") or {}
        datos["acc_val"] = None
        datos["extra"] = {"ventana": self.ventana, "horizonte": self.horizonte,
                          "columna": self.meta.get("columna"),
                          "mae": round(metricas.get("mae", 0), 2),
                          "mape": round(metricas.get("mape", 0) * 100, 1),
                          "mae_ingenuo": round(metricas.get("mae_ingenuo", 0), 2),
                          "serie": self._serie_guardada()}
        return datos

    def _serie_guardada(self) -> list[float]:
        """Los últimos valores del CSV con el que se entrenó, para poder dibujar."""
        try:
            from tareas.series import _leer_serie
            valores, _ = _leer_serie(Path(self.cfg.datos.ruta),
                                     self.meta.get("columna") or self.cfg.series.columna, "")
            return [round(v, 2) for v in valores[-120:]]
        except SystemExit:
            return []

    @torch.no_grad()
    def predecir(self, valores: list[float]) -> dict:
        self.asegurar()
        if len(valores) < self.ventana:
            raise ValueError(f"Hacen falta al menos {self.ventana} valores, hay {len(valores)}")
        ventana = np.array(valores[-self.ventana:], dtype=np.float32)
        normal = (ventana - self.media) / (self.desv or 1)
        salida = self.modelo(torch.tensor(normal).unsqueeze(0).to(self.dispositivo))
        prediccion = salida[0].cpu().numpy() * self.desv + self.media
        return {"prediccion": [round(float(v), 2) for v in prediccion],
                "ventana": [round(float(v), 2) for v in ventana],
                "horizonte": self.horizonte}


# ===========================================================================
# Entidades (NER) y búsqueda semántica
# ===========================================================================

class ServicioNER(Servicio):
    """Encuentra nombres, lugares y organizaciones en un texto. Sin entrenar, el modelo
    base ya reconoce PER/LOC/ORG; entrenado, reconoce además las entidades tuyas."""

    def cargar(self) -> None:
        import copy

        from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                                  pipeline)

        from nucleo.config import Config as _Config, DEFECTOS

        if self.vista.entrenada():
            ckpt = torch.load(self.vista.ruta_modelo, map_location="cpu", weights_only=False)
            self.cfg = _Config(ckpt["config"])
            base = ckpt.get("modelo_base", self.cfg.ner.modelo_base)
            etiquetas = ckpt.get("etiquetas", [])
            modelo = AutoModelForTokenClassification.from_pretrained(
                base, num_labels=len(etiquetas), ignore_mismatched_sizes=True)
            modelo.load_state_dict(ckpt["state_dict"])
            modelo.config.id2label = dict(enumerate(etiquetas))
        else:
            self.cfg = _Config(copy.deepcopy(DEFECTOS))
            ckpt, base = None, self.cfg.ner.modelo_base
            modelo = AutoModelForTokenClassification.from_pretrained(base)

        self.tokenizador = AutoTokenizer.from_pretrained(base)
        self.modelo = modelo.eval().to(self.dispositivo)
        self.tuberia = pipeline("token-classification", model=self.modelo,
                                tokenizer=self.tokenizador, aggregation_strategy="simple",
                                device=-1 if self.dispositivo == "cpu" else 0)
        self.meta = {"arquitectura": base, "clases": list(modelo.config.id2label.values()),
                     "fecha": (ckpt or {}).get("fecha"),
                     "metricas": (ckpt or {}).get("metricas", {})}

    def info(self) -> dict:
        datos = super().info()
        metricas = self.meta.get("metricas") or {}
        datos["acc_val"] = metricas.get("f1")
        datos["extra"] = {"sin_entrenar": not self.vista.entrenada(),
                          "tipos": sorted({e.split("-")[-1] for e in self.meta["clases"]
                                           if e != "O"})}
        return datos

    def predecir(self, texto: str) -> dict:
        import time
        self.asegurar()
        inicio = time.perf_counter()
        crudo = self.tuberia(texto[:4000])
        ms = (time.perf_counter() - inicio) * 1000

        entidades = [{"texto": e["word"], "tipo": e["entity_group"],
                      "inicio": int(e["start"]), "fin": int(e["end"]),
                      "confianza": float(e["score"])} for e in crudo]
        por_tipo = {}
        for e in entidades:
            por_tipo.setdefault(e["tipo"], []).append(e["texto"])
        return {"entidades": entidades, "por_tipo": por_tipo, "ms": ms,
                "caracteres": len(texto)}


def _trocear(texto: str, tamano: int, solape: int) -> list[str]:
    """Trocea respetando los párrafos.

    Cortar cada N caracteres a ciegas parte las ideas por la mitad y luego la búsqueda
    devuelve el fragmento equivocado. Se agrupa por párrafos y solo se subdivide el que
    se pase de largo.
    """
    parrafos = [" ".join(p.split()) for p in texto.split("\n\n")]
    parrafos = [p for p in parrafos if len(p) > 25]

    trozos, actual = [], ""
    for parrafo in parrafos:
        if len(parrafo) > tamano * 1.6:           # párrafo largo: se parte con solape
            if actual:
                trozos.append(actual)
                actual = ""
            i = 0
            while i < len(parrafo):
                trozos.append(parrafo[i:i + tamano])
                i += max(1, tamano - solape)
        elif len(actual) + len(parrafo) + 1 <= tamano:
            actual = f"{actual} {parrafo}".strip()
        else:
            if actual:
                trozos.append(actual)
            actual = parrafo
    if actual:
        trozos.append(actual)
    return [t for t in trozos if len(t.strip()) > 40]


class ServicioBusqueda(Servicio):
    """Búsqueda por significado: encuentra el fragmento que responde a la pregunta,
    aunque no compartan ni una palabra. Es la base de un asistente sobre documentos."""

    def __init__(self, vista):
        super().__init__(vista)
        self.indice_ruta = RAIZ / "modelos" / "indice_semantico.npz"
        self.documentos: list[dict] = []
        self.vectores = None

    def cargar(self) -> None:
        import copy

        from transformers import AutoModel, AutoTokenizer

        from nucleo.config import Config as _Config, DEFECTOS

        self.cfg = _Config(copy.deepcopy(DEFECTOS))
        base = self.cfg.busqueda.modelo_base
        self.tokenizador = AutoTokenizer.from_pretrained(base)
        self.modelo = AutoModel.from_pretrained(base).eval().to(self.dispositivo)
        self.meta = {"arquitectura": base, "clases": [], "fecha": "preentrenado",
                     "metricas": {}}
        self._cargar_indice()

    def _cargar_indice(self) -> None:
        if not self.indice_ruta.exists():
            self.documentos, self.vectores = [], None
            return
        datos = np.load(self.indice_ruta, allow_pickle=False)
        self.vectores = datos["vectores"]
        self.documentos = [{"texto": str(t), "origen": str(o)}
                           for t, o in zip(datos["textos"], datos["origenes"])]

    def info(self) -> dict:
        datos = super().info()
        datos["acc_val"] = None
        datos["extra"] = {"sin_entrenar": True, "fragmentos": len(self.documentos),
                          "origenes": sorted({d["origen"] for d in self.documentos})}
        return datos

    @torch.no_grad()
    def _vectorizar(self, textos: list[str], prefijo: str) -> np.ndarray:
        self.asegurar()
        salida = []
        for comienzo in range(0, len(textos), 16):
            lote = [f"{prefijo}: {t}" for t in textos[comienzo:comienzo + 16]]
            entradas = self.tokenizador(lote, padding=True, truncation=True,
                                        max_length=512, return_tensors="pt").to(self.dispositivo)
            estados = self.modelo(**entradas).last_hidden_state
            mascara = entradas["attention_mask"].unsqueeze(-1).float()
            # Media ponderada por la máscara: ignora el relleno
            medio = (estados * mascara).sum(1) / mascara.sum(1).clamp(min=1e-9)
            salida.append(torch.nn.functional.normalize(medio, dim=1).cpu().numpy())
        return np.vstack(salida)

    def indexar(self, texto: str, origen: str) -> dict:
        """Trocea el documento y guarda el vector de cada fragmento."""
        b = self.cfg.busqueda
        trozos = _trocear(texto, b.trozo, b.solape)
        if not trozos:
            raise ValueError("El documento es demasiado corto")

        nuevos = self._vectorizar(trozos, "passage")
        self.documentos = [d for d in self.documentos if d["origen"] != origen]
        if self.vectores is not None and len(self.documentos) != len(self.vectores):
            conservar = [i for i, d in enumerate(self._todos_origenes()) if d != origen]
            self.vectores = self.vectores[conservar] if conservar else None
        self.documentos += [{"texto": t, "origen": origen} for t in trozos]
        self.vectores = nuevos if self.vectores is None else np.vstack([self.vectores, nuevos])
        self._guardar()
        return {"origen": origen, "fragmentos": len(trozos), "total": len(self.documentos)}

    def _todos_origenes(self) -> list[str]:
        return [d["origen"] for d in self.documentos]

    def _guardar(self) -> None:
        self.indice_ruta.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.indice_ruta, vectores=self.vectores,
                 textos=np.array([d["texto"] for d in self.documentos]),
                 origenes=np.array([d["origen"] for d in self.documentos]))

    def buscar(self, pregunta: str, cuantos: int | None = None) -> dict:
        import time
        self.asegurar()
        if self.vectores is None or not len(self.documentos):
            return {"resultados": [], "detalle": "no hay documentos indexados"}
        inicio = time.perf_counter()
        vector = self._vectorizar([pregunta], "query")[0]
        similitudes = self.vectores @ vector
        orden = np.argsort(-similitudes)[:cuantos or self.cfg.busqueda.resultados]
        return {"resultados": [{"texto": self.documentos[int(i)]["texto"],
                                "origen": self.documentos[int(i)]["origen"],
                                "similitud": float(similitudes[int(i)])} for i in orden],
                "ms": (time.perf_counter() - inicio) * 1000,
                "fragmentos": len(self.documentos)}

    def olvidar_documento(self, origen: str) -> dict:
        conservar = [i for i, d in enumerate(self.documentos) if d["origen"] != origen]
        self.documentos = [self.documentos[i] for i in conservar]
        self.vectores = self.vectores[conservar] if conservar else None
        self._guardar()
        return {"fragmentos": len(self.documentos)}


# ===========================================================================
# Detección de objetos sin entrenar (COCO) y seguimiento
# ===========================================================================

COCO = [
    "__fondo__", "persona", "bicicleta", "coche", "moto", "avión", "autobús", "tren",
    "camión", "barco", "semáforo", "boca de incendios", "N/A", "señal de stop",
    "parquímetro", "banco", "pájaro", "gato", "perro", "caballo", "oveja", "vaca",
    "elefante", "oso", "cebra", "jirafa", "N/A", "mochila", "paraguas", "N/A", "N/A",
    "bolso", "corbata", "maleta", "frisbee", "esquís", "tabla de snow", "pelota",
    "cometa", "bate", "guante", "monopatín", "tabla de surf", "raqueta", "botella",
    "N/A", "copa", "taza", "tenedor", "cuchillo", "cuchara", "cuenco", "plátano",
    "manzana", "sándwich", "naranja", "brócoli", "zanahoria", "perrito caliente",
    "pizza", "dónut", "pastel", "silla", "sofá", "planta", "cama", "N/A", "mesa",
    "N/A", "N/A", "inodoro", "N/A", "televisor", "portátil", "ratón", "mando",
    "teclado", "teléfono", "microondas", "horno", "tostadora", "fregadero", "nevera",
    "N/A", "libro", "reloj", "jarrón", "tijeras", "peluche", "secador", "cepillo",
]


class ServicioObjetosCOCO(Servicio):
    """Detector ya entrenado con 80 objetos cotidianos: personas, vehículos, mobiliario.

    No hay que entrenar nada. La tarea `vision_deteccion` es para lo contrario: enseñarle
    objetos que este no conoce (un casco concreto, un formulario, una pieza).
    """

    def cargar(self) -> None:
        from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                                  fasterrcnn_resnet50_fpn_v2)
        pesos = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.modelo = fasterrcnn_resnet50_fpn_v2(weights=pesos).eval().to(self.dispositivo)
        self.meta = {"arquitectura": "fasterrcnn_resnet50_fpn_v2 (COCO)", "clases": COCO,
                     "fecha": "preentrenado", "metricas": {}}

    def info(self) -> dict:
        datos = super().info()
        datos["acc_val"] = None
        datos["extra"] = {"sin_entrenar": True, "clases": len(COCO) - 1}
        return datos

    @torch.no_grad()
    def predecir(self, imagen_bytes: bytes, umbral: float = 0.6,
                 solo: str | None = None) -> dict:
        import torchvision.transforms.functional as TF
        self.asegurar()
        imagen = self.abrir(imagen_bytes)
        x = TF.to_tensor(imagen).to(self.dispositivo)
        salida = self.modelo([x])[0]

        objetos = []
        for caja, etiqueta, score in zip(salida["boxes"].cpu().numpy(),
                                         salida["labels"].cpu().numpy(),
                                         salida["scores"].cpu().numpy()):
            if score < umbral:
                continue
            nombre = COCO[int(etiqueta)] if int(etiqueta) < len(COCO) else str(etiqueta)
            if solo and nombre != solo:
                continue
            objetos.append({
                "caja": {"x": float(caja[0]), "y": float(caja[1]),
                         "ancho": float(caja[2] - caja[0]), "alto": float(caja[3] - caja[1])},
                "clase": nombre, "indice": int(etiqueta), "confianza": float(score),
            })
        return {"objetos": objetos, "ancho": imagen.width, "alto": imagen.height}


# ===========================================================================
# Generación de imágenes
# ===========================================================================

class ServicioGeneracion(Servicio):
    """Genera imágenes desde texto. Sin entrenar usa el modelo base; con un LoRA
    entrenado, además sabe dibujar el sujeto que le enseñaste."""

    def cargar(self) -> None:
        import copy

        from diffusers import StableDiffusionPipeline

        from nucleo.config import Config as _Config, DEFECTOS

        if self.vista.entrenada():
            ckpt = torch.load(self.vista.ruta_modelo, map_location="cpu", weights_only=False)
            self.cfg = _Config(ckpt["config"])
            base = ckpt.get("modelo_base", self.cfg.generacion.modelo_base)
        else:
            self.cfg = _Config(copy.deepcopy(DEFECTOS))
            ckpt, base = None, self.cfg.generacion.modelo_base

        self.tuberia = StableDiffusionPipeline.from_pretrained(
            base, torch_dtype=torch.float16 if self.dispositivo == "cuda" else torch.float32,
            safety_checker=None, requires_safety_checker=False)
        self.tuberia.set_progress_bar_config(disable=True)
        self.tuberia.to(self.dispositivo)

        if ckpt is not None:
            self._aplicar_lora(ckpt)
        self.modelo = self.tuberia.unet
        self.meta = {"arquitectura": base, "clases": [],
                     "fecha": (ckpt or {}).get("fecha"),
                     "metricas": (ckpt or {}).get("metricas", {}),
                     "instancia": (ckpt or {}).get("instancia", "")}

    def _aplicar_lora(self, ckpt) -> None:
        from peft import LoraConfig, get_peft_model
        g = self.cfg.generacion
        unet = get_peft_model(self.tuberia.unet, LoraConfig(
            r=g.lora_r, lora_alpha=g.lora_alpha, init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
        faltan, sobran = unet.load_state_dict(ckpt["state_dict"], strict=False)
        if sobran:
            print(f"AVISO: {len(sobran)} pesos del LoRA no encajan en la U-Net")
        self.tuberia.unet = unet

    def info(self) -> dict:
        datos = super().info()
        datos["acc_val"] = None
        datos["extra"] = {"sin_entrenar": not self.vista.entrenada(),
                          "instancia": self.meta.get("instancia", ""),
                          "resolucion": self.cfg.generacion.resolucion,
                          "pasos": self.cfg.generacion.pasos,
                          "guia": self.cfg.generacion.guia}
        return datos

    @torch.no_grad()
    def generar(self, prompt: str, pasos: int | None = None, guia: float | None = None,
                negativo: str | None = None, semilla: int | None = None,
                lado: int | None = None) -> dict:
        import time
        self.asegurar()
        g = self.cfg.generacion
        lado = lado or g.resolucion
        generador = None
        if semilla is not None:
            generador = torch.Generator(device=self.dispositivo).manual_seed(int(semilla))

        inicio = time.perf_counter()
        salida = self.tuberia(
            prompt=prompt[:400],
            negative_prompt=(negativo if negativo is not None else g.negativo) or None,
            num_inference_steps=int(pasos or g.pasos),
            guidance_scale=float(guia if guia is not None else g.guia),
            height=lado, width=lado, generator=generador)
        ms = (time.perf_counter() - inicio) * 1000

        return {"png": _png(salida.images[0]), "ms": ms, "pasos": int(pasos or g.pasos),
                "semilla": semilla, "lado": lado}


# ===========================================================================
# Transcripción
# ===========================================================================

class ServicioTranscripcion(Servicio):
    """Voz a texto. Sin modelo propio usa Whisper tal cual, que ya funciona bien."""

    def cargar(self) -> None:
        import copy

        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        from nucleo.config import Config as _Config, DEFECTOS

        if self.vista.entrenada():
            ckpt = torch.load(self.vista.ruta_modelo, map_location="cpu", weights_only=False)
            self.cfg = _Config(ckpt["config"])
            base = ckpt.get("modelo_base", self.cfg.transcripcion.modelo_base)
        else:
            self.cfg = _Config(copy.deepcopy(DEFECTOS))
            ckpt, base = None, self.cfg.transcripcion.modelo_base

        self.procesador = WhisperProcessor.from_pretrained(
            base, language=self.cfg.transcripcion.idioma, task="transcribe")
        self.modelo = WhisperForConditionalGeneration.from_pretrained(base)
        if ckpt is not None:
            self.modelo.load_state_dict(ckpt["state_dict"])
        self.modelo.eval().to(self.dispositivo)
        self.meta = {"arquitectura": base, "clases": [],
                     "fecha": (ckpt or {}).get("fecha"),
                     "metricas": (ckpt or {}).get("metricas", {})}

    def info(self) -> dict:
        datos = super().info()
        metricas = self.meta.get("metricas") or {}
        datos["acc_val"] = None
        datos["extra"] = {"sin_entrenar": not self.vista.entrenada(),
                          "idioma": self.cfg.transcripcion.idioma,
                          "wer": round(metricas.get("wer", 0), 4) if metricas else None}
        return datos

    @torch.no_grad()
    def transcribir(self, audio_bytes: bytes, nombre: str = "audio.wav",
                    idioma: str | None = None) -> dict:
        import tempfile
        import time

        from tareas.audio_clasificacion.carga import cargar_audio

        self.asegurar()
        sufijo = Path(nombre).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=sufijo, delete=False) as tmp:
            tmp.write(audio_bytes)
            ruta = Path(tmp.name)
        try:
            onda, sr = cargar_audio(ruta)
            objetivo = self.cfg.transcripcion.sr
            if sr != objetivo:
                import torchaudio
                onda = torchaudio.functional.resample(onda, sr, objetivo)
            duracion = onda.shape[-1] / objetivo
            muestras = onda.squeeze(0).numpy()

            inicio = time.perf_counter()
            textos = []
            # Whisper trabaja en ventanas de 30 s: los audios largos se trocean
            ventana = int(objetivo * 30)
            for comienzo in range(0, max(1, len(muestras)), ventana):
                trozo = muestras[comienzo:comienzo + ventana]
                if len(trozo) < objetivo * 0.2:
                    continue
                rasgos = self.procesador.feature_extractor(
                    trozo, sampling_rate=objetivo, return_tensors="pt"
                ).input_features.to(self.dispositivo)
                ids = self.modelo.generate(
                    rasgos, max_new_tokens=220,
                    language=idioma or self.cfg.transcripcion.idioma, task="transcribe")
                textos.append(self.procesador.batch_decode(
                    ids, skip_special_tokens=True)[0].strip())
            ms = (time.perf_counter() - inicio) * 1000
        finally:
            ruta.unlink(missing_ok=True)

        return {"texto": " ".join(t for t in textos if t), "duracion": round(duracion, 2),
                "ms": ms, "idioma": idioma or self.cfg.transcripcion.idioma}


# ===========================================================================
# Síntesis y clonación de voz
# ===========================================================================

class ServicioVoz(Servicio):
    """Convierte texto en habla. Sin entrenar nada usa el modelo base; si has ajustado
    una voz, usa la tuya. El timbre lo decide un vector de hablante."""

    def __init__(self, vista):
        super().__init__(vista)
        self.dir_voces = RAIZ / "modelos" / "voces"
        self.vocoder = None

    def cargar(self) -> None:
        from transformers import (SpeechT5ForTextToSpeech, SpeechT5HifiGan,
                                  SpeechT5Processor)

        from nucleo.config import Config as _Config, DEFECTOS
        import copy

        if self.vista.entrenada():
            ckpt = torch.load(self.vista.ruta_modelo, map_location="cpu", weights_only=False)
            self.cfg = _Config(ckpt["config"])
            base = ckpt.get("modelo_base", self.cfg.voz.modelo_base)
        else:
            # Sin entrenamiento propio: se usa el modelo base tal cual
            self.cfg = _Config(copy.deepcopy(DEFECTOS))
            ckpt, base = None, self.cfg.voz.modelo_base

        self.procesador = SpeechT5Processor.from_pretrained(base)
        self.modelo = SpeechT5ForTextToSpeech.from_pretrained(base)
        if ckpt is not None:
            self.modelo.load_state_dict(ckpt["state_dict"])
        self.modelo.eval().to(self.dispositivo)
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.cfg.voz.vocoder).eval().to(
            self.dispositivo)
        self.meta = {"arquitectura": base, "clases": [],
                     "fecha": (ckpt or {}).get("fecha"), "metricas": (ckpt or {}).get("metricas", {})}

    def info(self) -> dict:
        datos = super().info()
        datos["acc_val"] = None
        datos["extra"] = {"sin_entrenar": not self.vista.entrenada(),
                          "voces_propias": self.listar_voces(),
                          "banco": self.cfg.voz.banco_voces,
                          "sr": self.cfg.voz.sr}
        return datos

    def listar_voces(self) -> list[str]:
        if not self.dir_voces.exists():
            return []
        return sorted(p.stem for p in self.dir_voces.glob("*.npy"))

    def _vector(self, voz: str | None) -> torch.Tensor:
        from tareas.voz_sintesis import voz_del_banco
        if voz and (self.dir_voces / f"{voz}.npy").exists():
            return torch.tensor(np.load(self.dir_voces / f"{voz}.npy"),
                                dtype=torch.float32).reshape(512)
        indice = int(voz) if (voz or "").isdigit() else self.cfg.voz.voz_por_defecto
        return voz_del_banco(self.cfg.voz.banco_voces, indice)

    @torch.no_grad()
    def hablar(self, texto: str, voz: str | None = None) -> dict:
        import time
        self.asegurar()
        entradas = self.procesador(text=texto[:600], return_tensors="pt").to(self.dispositivo)
        vector = self._vector(voz).unsqueeze(0).to(self.dispositivo)

        inicio = time.perf_counter()
        onda = self.modelo.generate_speech(entradas["input_ids"], vector, vocoder=self.vocoder)
        ms = (time.perf_counter() - inicio) * 1000

        return {"wav": _a_wav(onda.cpu().numpy(), self.cfg.voz.sr),
                "segundos": round(len(onda) / self.cfg.voz.sr, 2), "ms": ms,
                "voz": voz or str(self.cfg.voz.voz_por_defecto)}

    def clonar(self, nombre: str, audio_bytes: bytes, extension: str = ".wav") -> dict:
        """Extrae el timbre de una grabación y lo guarda como una voz más."""
        import tempfile

        from tareas.voz_sintesis import extraer_vector

        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
            tmp.write(audio_bytes)
            ruta = Path(tmp.name)
        try:
            vector = extraer_vector(ruta, self.cfg.voz.sr)
        finally:
            ruta.unlink(missing_ok=True)

        self.dir_voces.mkdir(parents=True, exist_ok=True)
        np.save(self.dir_voces / f"{nombre}.npy", vector)
        return {"nombre": nombre, "voces": self.listar_voces()}

    def olvidar_voz(self, nombre: str) -> dict:
        archivo = self.dir_voces / f"{nombre}.npy"
        if not archivo.exists():
            raise ValueError(f"No existe la voz '{nombre}'")
        archivo.unlink()
        return {"voces": self.listar_voces()}


def _a_wav(muestras: np.ndarray, sr: int) -> bytes:
    """WAV PCM 16 bits sin depender de soundfile."""
    import struct
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        datos = np.clip(muestras, -1, 1)
        f.writeframes((datos * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


# ===========================================================================
# Texto y modelo de lenguaje
# ===========================================================================

class ServicioTexto(Servicio):
    def cargar(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        ckpt = torch.load(self.vista.ruta_modelo, map_location=self.dispositivo,
                          weights_only=False)
        self.cfg = Config(ckpt["config"])
        base = ckpt.get("modelo_base", self.cfg.texto.modelo_base)
        self.clases = ckpt["clases"]
        self.tokenizador = AutoTokenizer.from_pretrained(base)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=len(self.clases))
        self.modelo.load_state_dict(ckpt["state_dict"])
        self.modelo.eval().to(self.dispositivo)
        self.meta = {"arquitectura": base, "clases": self.clases,
                     "fecha": ckpt.get("fecha"), "metricas": ckpt.get("metricas", {}),
                     "temperatura": (ckpt.get("metricas") or {}).get("temperatura", 1.0)}

    @torch.no_grad()
    def predecir(self, texto: str) -> dict:
        self.asegurar()
        entradas = self.tokenizador(texto, truncation=True,
                                    max_length=self.cfg.texto.longitud_max,
                                    return_tensors="pt").to(self.dispositivo)
        logits = self.modelo(**entradas).logits
        temperatura = float(self.meta.get("temperatura") or 1.0)
        probs = (logits / temperatura).softmax(1)[0].cpu().numpy()
        indice = int(probs.argmax())
        return {"etiqueta": self.clases[indice], "confianza": float(probs[indice]),
                "probabilidades": {c: float(p) for c, p in zip(self.clases, probs)},
                "tokens": int(entradas["input_ids"].shape[1])}


class ServicioLLM(Servicio):
    def cargar(self) -> None:
        from chatear import cargar as cargar_llm
        self.modelo, self.tokenizador, self.cfg, self.dispositivo = cargar_llm(
            self.vista.ruta_modelo.parent, sin_adaptador=False)
        ckpt = torch.load(self.vista.ruta_modelo, map_location="cpu", weights_only=False)
        self.meta = {"arquitectura": ckpt.get("modelo_base", self.cfg.llm.modelo_base),
                     "clases": [], "fecha": ckpt.get("fecha"),
                     "metricas": ckpt.get("metricas", {})}

    def info(self) -> dict:
        datos = super().info()
        metricas = self.meta.get("metricas") or {}
        datos["acc_val"] = None
        datos["extra"] = {"perplejidad": metricas.get("perplejidad")}
        return datos

    def responder(self, instruccion: str, entrada: str = "", tokens: int = 200,
                  temperatura: float = 0.7) -> dict:
        self.asegurar()
        from chatear import responder as generar
        import time
        inicio = time.perf_counter()
        texto = generar(self.modelo, self.tokenizador, self.cfg, self.dispositivo,
                        instruccion, entrada, tokens, temperatura)
        return {"respuesta": texto, "ms": (time.perf_counter() - inicio) * 1000}


# ===========================================================================

CONSTRUCTORES = {
    "genero": lambda v: ServicioImagen(v),
    "atributos": lambda v: ServicioImagen(v),
    "antispoofing": lambda v: ServicioImagen(v),
    "rostros": lambda v: ServicioRostros(v),
    "deteccion": lambda v: ServicioDeteccion(v),
    "segmentacion": lambda v: ServicioSegmentacion(v),
    "superresolucion": lambda v: ServicioSuperResolucion(v),
    "audio": lambda v: ServicioAudio(v),
    "voz": lambda v: ServicioVoz(v),
    "transcripcion": lambda v: ServicioTranscripcion(v),
    "generacion": lambda v: ServicioGeneracion(v),
    "pose": lambda v: ServicioPose(v),
    "profundidad": lambda v: ServicioProfundidad(v),
    "ocr": lambda v: ServicioOCR(v),
    "ner": lambda v: ServicioNER(v),
    "tabular": lambda v: ServicioTabular(v),
    "series": lambda v: ServicioSeries(v),
    "anomalias": lambda v: ServicioAnomalias(v),
    "busqueda": lambda v: ServicioBusqueda(v),
    "imagenes": lambda v: ServicioImagenesTexto(v),
    "seguimiento": lambda v: ServicioObjetosCOCO(v),
    "texto": lambda v: ServicioTexto(v),
    "llm": lambda v: ServicioLLM(v),
}

# Cuántos modelos se mantienen cargados a la vez. Entre todas las vistas hay varios GB
# en modelos; sin este límite, recorrer el panel entero agota la memoria y el proceso
# muere sin dejar rastro. Se desaloja el que lleve más tiempo sin usarse.
MAX_CARGADOS = int(os.environ.get("PANEL_MAX_MODELOS", "3"))

_CACHE: "OrderedDict[str, Servicio]" = OrderedDict()


_COMPARADORES: "OrderedDict[str, Servicio]" = OrderedDict()


def comparador(experimento: str) -> Servicio:
    """Servicio de clasificación de imagen para un experimento cualquiera.

    Permite pasar la misma foto por varios modelos y ver la diferencia — por ejemplo
    el de tres épocas frente al de veinticinco.
    """
    if experimento in _COMPARADORES:
        _COMPARADORES.move_to_end(experimento)
    else:
        while len(_COMPARADORES) >= 4:      # comparar más de 4 a la vez no aporta
            _, viejo = _COMPARADORES.popitem(last=False)
            viejo.liberar()
        from servidor.vistas import Vista
        vista = Vista(f"comparar_{experimento}", experimento, "🔬", "",
                      "imagen_clasificacion", experimento, "imagen", "")
        _COMPARADORES[experimento] = ServicioImagen(vista)
    return _COMPARADORES[experimento].asegurar()


def obtener(vista) -> Servicio:
    """Servicio de una vista, con desalojo por uso para no quedarse sin memoria."""
    if vista.slug in _CACHE:
        _CACHE.move_to_end(vista.slug)
    else:
        _liberar_sitio()
        _CACHE[vista.slug] = CONSTRUCTORES[vista.slug](vista)
    return _CACHE[vista.slug].asegurar()


def _liberar_sitio() -> None:
    while len(_CACHE) >= MAX_CARGADOS:
        slug, servicio = _CACHE.popitem(last=False)
        print(f"[panel] descargando el modelo de '{slug}' para hacer sitio")
        servicio.liberar()


def cargados() -> list[str]:
    return list(_CACHE)


def olvidar(slug: str) -> None:
    servicio = _CACHE.pop(slug, None)
    if servicio is not None:
        servicio.liberar()


def vaciar() -> int:
    cuantos = len(_CACHE)
    for slug in list(_CACHE):
        olvidar(slug)
    return cuantos

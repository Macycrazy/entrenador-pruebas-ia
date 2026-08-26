#!/usr/bin/env python3
"""Galería de rostros: inscribe personas conocidas y reconócelas después.

    # 1. inscribir: una carpeta por persona con una o varias fotos
    python galeria_rostros.py inscribir --modelo experimentos/rostro_id \\
        --fotos plantilla/ --salida modelos/galeria.npz

    # 2. identificar una foto suelta
    python galeria_rostros.py identificar --galeria modelos/galeria.npz --foto entrada.jpg

    # 3. verificar si dos fotos son la misma persona
    python galeria_rostros.py verificar --galeria modelos/galeria.npz --foto a.jpg --contra b.jpg

Es el modo de uso real en un control de acceso: el modelo no se reentrena para
añadir a alguien; basta con inscribir sus fotos, que se guardan como un vector.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

import numpy as np
import torch
from PIL import Image

from comun.rostros import DetectorRostros, recortar_rostro
from nucleo.carga import cargar_modelo, metadatos
from nucleo.config import Config
from tareas.imagen_clasificacion import aumentos

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Galería de rostros")
    sub = p.add_subparsers(dest="accion", required=True)

    inscribir = sub.add_parser("inscribir", help="Crear la galería desde carpetas de personas")
    inscribir.add_argument("--modelo", required=True, help="Experimento o .pt entrenado")
    inscribir.add_argument("--fotos", type=Path, required=True,
                           help="Carpeta con una subcarpeta por persona")
    inscribir.add_argument("--salida", type=Path, default=RAIZ / "modelos" / "galeria.npz")
    inscribir.add_argument("--sin-detector", action="store_true",
                           help="No recortar el rostro (las fotos ya vienen recortadas)")

    identificar = sub.add_parser("identificar", help="¿Quién es la persona de esta foto?")
    identificar.add_argument("--galeria", type=Path, required=True)
    identificar.add_argument("--foto", type=Path, required=True)
    identificar.add_argument("--umbral", type=float, help="Coseno mínimo para aceptar")
    identificar.add_argument("--top", type=int, default=3)

    verificar = sub.add_parser("verificar", help="¿Son la misma persona estas dos fotos?")
    verificar.add_argument("--galeria", type=Path, required=True)
    verificar.add_argument("--foto", type=Path, required=True)
    verificar.add_argument("--contra", type=Path, required=True)
    verificar.add_argument("--umbral", type=float)
    return p.parse_args()


def _backbone_suelto(especificacion: str, dispositivo: str):
    """Usa un backbone preentrenado sin haber entrenado nada.

    `rostro:facenet` ya viene entrenado con millones de caras (VGGFace2), así que sirve
    tal cual para inscribir y reconocer. Es la forma de tener reconocimiento facial el
    primer día, antes de entrenar con fotos propias.
    """
    from nucleo.config import DEFECTOS
    from tareas.rostro_identificacion.modelos import crear_modelo

    cfg = Config(json.loads(json.dumps(DEFECTOS)))
    cfg["modelo"]["arquitectura"] = especificacion
    cfg["rostros"]["dim_embedding"] = 0        # sin cuello: el backbone ya está entrenado
    cfg["datos"]["tam_img"] = 160
    modelo = crear_modelo(cfg, identidades=2).eval().to(dispositivo)
    meta = {"tam_img": 160, "arquitectura": especificacion, "clases": [],
            "metricas": {}, "tarea": "rostro_identificacion"}
    print(f"Backbone {especificacion} sin entrenar (embeddings del preentrenado)")
    return modelo, cfg, meta


class Extractor:
    """Convierte una foto en el vector que representa la cara."""

    def __init__(self, ruta_modelo, usar_detector: bool = True):
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        if str(ruta_modelo).startswith(("rostro:", "timm:")):
            self.modelo, self.cfg, self.meta = _backbone_suelto(str(ruta_modelo), dispositivo)
        else:
            self.modelo, ckpt = cargar_modelo(ruta_modelo, dispositivo)
            self.meta = metadatos(ckpt)
            self.cfg = Config(ckpt["config"])
        self.dispositivo = dispositivo
        self.transformacion = aumentos.validacion(self.cfg, self.meta["tam_img"])
        self.detector = DetectorRostros() if usar_detector else None
        # El umbral sale del entrenamiento (el coseno con el que se alcanzó el FAR
        # objetivo); si el modelo no viene de un entrenamiento, se usa el de la config.
        self.umbral = (self.meta.get("metricas") or {}).get(
            "umbral", self.cfg.rostros.umbral_similitud)

    @torch.no_grad()
    def vector(self, ruta: Path) -> np.ndarray | None:
        with Image.open(ruta) as bruta:
            imagen = bruta.convert("RGB")

        if self.detector is not None:
            import cv2
            bgr = cv2.cvtColor(np.asarray(imagen), cv2.COLOR_RGB2BGR)
            cajas = self.detector.detectar(bgr)
            if cajas:
                recorte = recortar_rostro(bgr, cajas[0])
                imagen = Image.fromarray(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))

        x = self.transformacion(imagen).unsqueeze(0).to(self.dispositivo)
        base = getattr(self.modelo, "_orig_mod", self.modelo)
        if hasattr(base, "embeddings"):
            emb = base.embeddings(x)
        else:                       # un clasificador normal: se usa su capa de rasgos
            salida = self.modelo(x, None, solo_logits=False)
            emb = torch.nn.functional.normalize(salida["rasgos"])
        return emb.squeeze(0).float().cpu().numpy()


def inscribir(args) -> None:
    extractor = Extractor(args.modelo, not args.sin_detector)
    nombres, vectores, conteos = [], [], []

    for carpeta in sorted(p for p in args.fotos.iterdir() if p.is_dir()):
        fotos = [f for f in sorted(carpeta.rglob("*")) if f.suffix.lower() in EXTENSIONES]
        embeddings = [v for v in (extractor.vector(f) for f in fotos) if v is not None]
        if not embeddings:
            print(f"  {carpeta.name}: sin fotos utilizables, se omite")
            continue
        # El centroide de varias fotos es mucho más estable que una sola.
        medio = np.mean(embeddings, axis=0)
        medio /= np.linalg.norm(medio) + 1e-9
        nombres.append(carpeta.name)
        vectores.append(medio)
        conteos.append(len(embeddings))
        print(f"  {carpeta.name}: {len(embeddings)} fotos")

    if not nombres:
        raise SystemExit(f"No se inscribió a nadie. ¿Hay subcarpetas por persona en {args.fotos}?")

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.salida, nombres=np.array(nombres), vectores=np.stack(vectores),
             conteos=np.array(conteos), modelo=str(args.modelo),
             umbral=float(extractor.umbral))
    print(f"\n{len(nombres)} personas inscritas en {args.salida}")
    print(f"Umbral sugerido (del entrenamiento): {extractor.umbral:.3f}")


def _cargar_galeria(ruta: Path):
    if not ruta.exists():
        raise SystemExit(f"No existe la galería {ruta}. Créala con la acción 'inscribir'.")
    datos = np.load(ruta, allow_pickle=False)
    return (list(datos["nombres"]), datos["vectores"], float(datos["umbral"]),
            str(datos["modelo"]))


def identificar(args) -> None:
    nombres, vectores, umbral, modelo = _cargar_galeria(args.galeria)
    umbral = args.umbral if args.umbral is not None else umbral
    extractor = Extractor(modelo)

    vector = extractor.vector(args.foto)
    if vector is None:
        raise SystemExit("No se pudo leer la foto")
    similitudes = vectores @ vector
    orden = np.argsort(-similitudes)[:args.top]

    mejor = orden[0]
    if similitudes[mejor] >= umbral:
        print(f"\n  → {nombres[mejor]}  (coseno {similitudes[mejor]:.3f}, umbral {umbral:.3f})")
    else:
        print(f"\n  → DESCONOCIDO  (el más parecido era {nombres[mejor]} con "
              f"{similitudes[mejor]:.3f}, por debajo del umbral {umbral:.3f})")

    print("\n  candidatos:")
    for indice in orden:
        print(f"    {nombres[indice]:<30} {similitudes[indice]:.3f}")


def verificar(args) -> None:
    _, _, umbral, modelo = _cargar_galeria(args.galeria)
    umbral = args.umbral if args.umbral is not None else umbral
    extractor = Extractor(modelo)
    a, b = extractor.vector(args.foto), extractor.vector(args.contra)
    similitud = float(a @ b)
    veredicto = "LA MISMA PERSONA" if similitud >= umbral else "PERSONAS DISTINTAS"
    print(f"\n  coseno {similitud:.3f} · umbral {umbral:.3f}  →  {veredicto}")


def main() -> None:
    args = argumentos()
    {"inscribir": inscribir, "identificar": identificar, "verificar": verificar}[args.accion](args)


if __name__ == "__main__":
    main()

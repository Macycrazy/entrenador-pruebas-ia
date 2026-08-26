#!/usr/bin/env python3
"""Evaluación completa de un modelo entrenado.

    python evaluar.py experimentos/calidad
    python evaluar.py experimentos/calidad --set datos.ruta=datos_otro --tta
    python evaluar.py experimentos/calidad --galeria   # HTML con los peores fallos

Da acierto global y balanceado, matriz de confusión, precisión/recall/F1 por clase,
AUC, calibración (ECE antes y después de temperature scaling) y, si el dataset trae
metadatos, el desglose por subgrupo — que es donde se ve el sesgo.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

import numpy as np
import torch

from nucleo import metricas as metricas_mod
from nucleo.bucle import preparar_dispositivo
from nucleo.carga import cargar_modelo
from nucleo.config import Config
from nucleo.tarea import crear_tarea


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evalúa un checkpoint")
    p.add_argument("experimento", type=Path, help="Carpeta del experimento o archivo .pt")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="CLAVE=VALOR")
    p.add_argument("--tta", action="store_true", help="Promediar con la imagen espejada")
    p.add_argument("--galeria", action="store_true", help="HTML con los errores más confiados")
    p.add_argument("--top", type=int, default=60, help="Cuántos errores mostrar en la galería")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = argumentos()
    modelo, ckpt = cargar_modelo(args.experimento)
    cfg = Config(ckpt["config"])
    for override in args.overrides:
        from nucleo.config import _aplicar
        _aplicar(cfg, override)
    if args.tta:
        cfg["evaluacion"]["tta"] = True

    dispositivo = preparar_dispositivo(cfg)
    modelo = modelo.to(dispositivo).eval()
    tarea = crear_tarea(cfg)
    _, loader_val, info = tarea.datos()
    criterio = tarea.criterio(info, dispositivo)
    evaluador = tarea.evaluador(info)

    print(f"\nModelo {ckpt.get('arquitectura')} · {ckpt.get('tam_img')}px · "
          f"entrenado {ckpt.get('fecha')}\n")

    probabilidades = []
    for lote in loader_val:
        paso = tarea.paso(modelo, lote, criterio, dispositivo, entrenando=False)
        logits = paso.logits
        if cfg.evaluacion.tta:
            espejo = tarea.paso(modelo, lote, criterio, dispositivo,
                                entrenando=False, espejo=True)
            logits = (logits.softmax(1) + espejo.logits.softmax(1)).log()
        evaluador.actualizar(logits, paso.objetivos, paso.perdida.item(), paso.subgrupos)
        probabilidades.append(logits.softmax(1).cpu().numpy())

    resumen = evaluador.resumen(calibrar=True, curvas=True)
    print(metricas_mod.formatear(resumen, info.clases))

    salida = args.experimento if args.experimento.is_dir() else args.experimento.parent
    (salida / "evaluacion.json").write_text(
        __import__("json").dumps(resumen, indent=2, ensure_ascii=False, default=str))
    print(f"\nInforme en {salida / 'evaluacion.json'}")

    if args.galeria and hasattr(tarea, "ds_val"):
        ruta = salida / "errores.html"
        generar_galeria(tarea.ds_val, np.concatenate(probabilidades), info.clases, ruta, args.top)
        print(f"Galería de fallos en {ruta}")


def generar_galeria(dataset, probs: np.ndarray, clases: list[str], destino: Path,
                    top: int) -> None:
    """Los errores más confiados son los que más enseñan sobre qué falla el modelo."""
    fallos = []
    for indice, muestra in enumerate(dataset.muestras[:len(probs)]):
        predicho = int(probs[indice].argmax())
        if predicho != muestra.etiqueta:
            fallos.append((float(probs[indice][predicho]), muestra, predicho))
    fallos.sort(key=lambda f: -f[0])

    tarjetas = "\n".join(
        f'<figure><img src="{html.escape(str(m.ruta.resolve().as_uri()))}" loading="lazy">'
        f'<figcaption>real <b>{html.escape(clases[m.etiqueta])}</b> · '
        f'predicho <b>{html.escape(clases[p])}</b> {c * 100:.0f}%</figcaption></figure>'
        for c, m, p in fallos[:top])

    destino.write_text(f"""<!doctype html><meta charset="utf-8">
<title>Errores del modelo</title>
<style>
 body{{background:#0b1020;color:#e8ecf8;font-family:system-ui;margin:24px}}
 h1{{font-size:18px}} .rejilla{{display:grid;gap:14px;
   grid-template-columns:repeat(auto-fill,minmax(160px,1fr))}}
 figure{{margin:0;background:#141a2e;border:1px solid #2a3355;border-radius:10px;padding:8px}}
 img{{width:100%;border-radius:6px;display:block}}
 figcaption{{font-size:11.5px;color:#98a3c4;margin-top:6px;line-height:1.4}}
</style>
<h1>{len(fallos)} errores · se muestran los {min(top, len(fallos))} más confiados</h1>
<div class="rejilla">{tarjetas}</div>""")


if __name__ == "__main__":
    main()

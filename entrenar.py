#!/usr/bin/env python3
"""Punto de entrada único de entrenamiento, para cualquier tarea.

    python entrenar.py --preset calidad
    python entrenar.py --config configs/genero.yaml --set entrenamiento.epocas=30
    python entrenar.py --tarea audio_clasificacion --config configs/voz.yaml
    python entrenar.py --buscar-lr                 # test de rango de learning rate
    python entrenar.py --kfold 5                   # entrena los 5 folds y promedia
    python entrenar.py --listar-opciones           # catálogo completo de opciones
    python entrenar.py --listar-modelos            # backbones disponibles
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from nucleo import config as config_mod
from nucleo.bucle import Entrenador
from nucleo.experimento import Experimento
from nucleo.tarea import TAREAS_CONOCIDAS, crear_tarea


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entrenamiento multitarea",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--config", type=Path, help="YAML con la configuración")
    p.add_argument("--preset", help=f"Preset de configs/ ({', '.join(config_mod.presets())})")
    p.add_argument("--tarea", choices=sorted(TAREAS_CONOCIDAS), help="Sobrescribe cfg.tarea")
    p.add_argument("--nombre", help="Nombre del experimento (carpeta de salida)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="CLAVE=VALOR",
                   help="Sobrescribe cualquier opción, ej. --set datos.batch=64")
    p.add_argument("--reanudar", help="Checkpoint o carpeta de experimento a continuar")
    p.add_argument("--buscar-lr", action="store_true", help="Solo buscar el learning rate")
    p.add_argument("--kfold", type=int, default=0, help="Entrena N folds y promedia resultados")
    p.add_argument("--listar-opciones", action="store_true")
    p.add_argument("--listar-modelos", action="store_true")
    return p.parse_args()


def main() -> None:
    args = argumentos()

    if args.listar_opciones:
        print(json.dumps(config_mod.DEFECTOS, indent=2, ensure_ascii=False))
        print("\nCada clave se puede cambiar con --set seccion.clave=valor "
              "o desde un YAML. Los comentarios del catálogo están en nucleo/config.py")
        return

    if args.listar_modelos:
        from tareas.imagen_clasificacion.modelos import listar_arquitecturas
        for familia, nombres in listar_arquitecturas().items():
            print(f"\n### {familia}  ({len(nombres)})")
            print(", ".join(nombres[:80]) + (" …" if len(nombres) > 80 else ""))
        return

    cfg = config_mod.cargar(args.config, args.preset, args.overrides)
    if args.tarea:
        cfg["tarea"] = args.tarea
    if args.nombre:
        cfg["nombre"] = args.nombre
    if args.reanudar:
        cfg["entrenamiento"]["reanudar"] = args.reanudar

    if args.buscar_lr:
        buscar_lr(cfg)
        return

    if args.kfold:
        validacion_cruzada(cfg, args.kfold)
        return

    entrenar(cfg)


def entrenar(cfg) -> dict:
    tarea = crear_tarea(cfg)
    experimento = Experimento(cfg)
    entrenador = Entrenador(cfg, tarea, experimento).preparar()
    resumen = entrenador.ejecutar()
    print(f"\nExperimento en {experimento.dir}")
    return resumen


def buscar_lr(cfg) -> None:
    """Sube el learning rate exponencialmente hasta que la pérdida explota."""
    from nucleo.bucle import preparar_dispositivo
    from nucleo.optimizadores import buscar_lr as buscar

    tarea = crear_tarea(cfg)
    dispositivo = preparar_dispositivo(cfg)
    loader_train, _, info = tarea.datos()
    modelo = tarea.modelo(info).to(dispositivo)
    criterio = tarea.criterio(info, dispositivo)

    print("Buscando learning rate…")
    resultado = buscar(modelo, loader_train, criterio, dispositivo, cfg)
    print(f"\nSugerido: {resultado['sugerido']:.2e}")
    print(f"Aplícalo con:  --set optimizador.lr={resultado['sugerido']:.2e}")
    for lr, perdida in zip(resultado["lrs"][::5], resultado["perdidas"][::5]):
        barra = "■" * int(min(40, perdida * 12))
        print(f"  {lr:.2e}  {perdida:.4f}  {barra}")


def validacion_cruzada(cfg, k: int) -> None:
    resultados = []
    nombre_base = cfg.nombre
    for fold in range(k):
        print(f"\n{'=' * 70}\nFOLD {fold + 1}/{k}\n{'=' * 70}")
        cfg["datos"]["kfold"] = k
        cfg["datos"]["fold"] = fold
        cfg["nombre"] = f"{nombre_base}_fold{fold}"
        resultados.append(entrenar(cfg))

    import statistics
    print(f"\n{'=' * 70}\nVALIDACIÓN CRUZADA {k} FOLDS\n{'=' * 70}")
    for metrica in ("acc", "acc_balanceada", "auc"):
        valores = [r[metrica] for r in resultados if metrica in r]
        if valores:
            desv = statistics.stdev(valores) if len(valores) > 1 else 0.0
            print(f"  {metrica:<16} {statistics.mean(valores):.4f} ± {desv:.4f}")


if __name__ == "__main__":
    main()

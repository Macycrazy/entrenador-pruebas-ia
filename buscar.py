#!/usr/bin/env python3
"""Búsqueda de hiperparámetros con Optuna (bayesiana) o aleatoria.

    python buscar.py --pruebas 20 --preset calidad --set entrenamiento.epocas=6
    python buscar.py --pruebas 12 --espacio espacio.json --metodo aleatorio

Cada prueba entrena de verdad con menos épocas y se queda con la combinación que
mejor validación da. El espacio por defecto cubre lo que más suele mover la aguja:
learning rate, weight decay, arquitectura, aumentaciones y regularización.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from nucleo import config as config_mod

ESPACIO = {
    "optimizador.lr": {"tipo": "log", "min": 5e-5, "max": 3e-3},
    "optimizador.wd": {"tipo": "log", "min": 1e-4, "max": 0.1},
    "optimizador.lr_backbone_factor": {"tipo": "float", "min": 0.05, "max": 1.0},
    "optimizador.nombre": {"tipo": "opcion", "valores": ["adamw", "sgd", "lion"]},
    "scheduler.nombre": {"tipo": "opcion", "valores": ["coseno", "onecycle"]},
    "modelo.dropout": {"tipo": "float", "min": 0.0, "max": 0.5},
    "perdida.suavizado": {"tipo": "float", "min": 0.0, "max": 0.15},
    "aumentos.politica": {"tipo": "opcion",
                          "valores": ["basica", "randaugment", "trivialaugment"]},
    "aumentos.mixup": {"tipo": "float", "min": 0.0, "max": 0.4},
    "aumentos.webcam": {"tipo": "float", "min": 0.0, "max": 0.6},
}


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Búsqueda de hiperparámetros")
    p.add_argument("--pruebas", type=int, default=20)
    p.add_argument("--config", type=Path)
    p.add_argument("--preset", default="rapido")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--espacio", type=Path, help="JSON con un espacio de búsqueda propio")
    p.add_argument("--metodo", default="optuna", choices=["optuna", "aleatorio"])
    p.add_argument("--metrica", default="acc_balanceada")
    return p.parse_args()


def main() -> None:
    args = argumentos()
    espacio = json.loads(args.espacio.read_text()) if args.espacio else ESPACIO
    base = config_mod.cargar(args.config, args.preset, args.overrides)
    print(f"Buscando sobre {len(espacio)} hiperparámetros · {args.pruebas} pruebas · "
          f"métrica {args.metrica}\n")

    if args.metodo == "optuna":
        mejor, historial = _con_optuna(base, espacio, args)
    else:
        mejor, historial = _aleatorio(base, espacio, args)

    print("\n" + "=" * 70)
    print("MEJOR COMBINACIÓN")
    for clave, valor in mejor["parametros"].items():
        print(f"  --set {clave}={valor}")
    print(f"  → {args.metrica} = {mejor['valor']:.4f}")

    destino = Path(base.salida.dir) / f"busqueda_{base.nombre}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({"mejor": mejor, "historial": historial},
                                  indent=2, ensure_ascii=False))
    print(f"\nHistorial completo en {destino}")


def _muestrear(espacio: dict, sugerir=None) -> dict:
    """Con Optuna usa `sugerir`; sin él, muestreo aleatorio."""
    salida = {}
    for clave, rango in espacio.items():
        if rango["tipo"] == "opcion":
            salida[clave] = sugerir.suggest_categorical(clave, rango["valores"]) if sugerir \
                else random.choice(rango["valores"])
        elif rango["tipo"] == "log":
            salida[clave] = sugerir.suggest_float(clave, rango["min"], rango["max"], log=True) \
                if sugerir else 10 ** random.uniform(
                    __import__("math").log10(rango["min"]), __import__("math").log10(rango["max"]))
        elif rango["tipo"] == "entero":
            salida[clave] = sugerir.suggest_int(clave, rango["min"], rango["max"]) if sugerir \
                else random.randint(rango["min"], rango["max"])
        else:
            salida[clave] = sugerir.suggest_float(clave, rango["min"], rango["max"]) if sugerir \
                else random.uniform(rango["min"], rango["max"])
    return salida


def _entrenar_con(base, parametros: dict, indice: int, metrica: str) -> float:
    import copy

    from entrenar import entrenar

    cfg = config_mod.Config(copy.deepcopy(dict(base)))
    cfg["nombre"] = f"{base.nombre}_prueba{indice:03d}"
    cfg["salida"]["tensorboard"] = False
    for clave, valor in parametros.items():
        config_mod._aplicar(cfg, f"{clave}={valor}")

    print(f"\n--- prueba {indice + 1} --- " +
          " ".join(f"{k.split('.')[-1]}={v:.4g}" if isinstance(v, float) else f"{k.split('.')[-1]}={v}"
                   for k, v in parametros.items()))
    try:
        resumen = entrenar(cfg)
    except Exception as error:  # noqa: BLE001 - una combinación mala no debe parar la búsqueda
        print(f"    prueba fallida: {error}")
        return 0.0
    return float(resumen.get(metrica, resumen.get("acc", 0.0)))


def _con_optuna(base, espacio, args):
    try:
        import optuna
    except ImportError:
        raise SystemExit("Necesita:  pip install optuna") from None

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    historial = []

    def objetivo(prueba):
        parametros = _muestrear(espacio, prueba)
        valor = _entrenar_con(base, parametros, prueba.number, args.metrica)
        historial.append({"parametros": parametros, "valor": valor})
        return valor

    estudio = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=base.semilla))
    estudio.optimize(objetivo, n_trials=args.pruebas)
    return {"parametros": estudio.best_params, "valor": estudio.best_value}, historial


def _aleatorio(base, espacio, args):
    random.seed(base.semilla)
    historial = []
    for i in range(args.pruebas):
        parametros = _muestrear(espacio)
        valor = _entrenar_con(base, parametros, i, args.metrica)
        historial.append({"parametros": parametros, "valor": valor})
    mejor = max(historial, key=lambda h: h["valor"])
    return mejor, historial


if __name__ == "__main__":
    main()

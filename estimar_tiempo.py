#!/usr/bin/env python3
"""Mide cuánto va a tardar un entrenamiento antes de lanzarlo.

Ejecuta unos pocos lotes reales (mismo modelo, mismos datos, misma precisión), mide
la velocidad y extrapola. Además comprueba si el cuello de botella es la GPU o la CPU
preparando los datos, que es el error más común: una tarjeta rápida esperando al
DataLoader.

    python estimar_tiempo.py --config configs/genero.yaml
    python estimar_tiempo.py --preset maxima --lotes 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

import torch

from nucleo import config as config_mod
from nucleo.bucle import preparar_dispositivo
from nucleo.optimizadores import SAM, crear_optimizador
from nucleo.tarea import crear_tarea


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estima la duración de un entrenamiento")
    p.add_argument("--config", type=Path)
    p.add_argument("--preset")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--tarea")
    p.add_argument("--lotes", type=int, default=20, help="Lotes cronometrados")
    p.add_argument("--calentamiento", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = argumentos()
    cfg = config_mod.cargar(args.config, args.preset, args.overrides)
    if args.tarea:
        cfg["tarea"] = args.tarea

    dispositivo = preparar_dispositivo(cfg)
    tarea = crear_tarea(cfg)
    loader_train, loader_val, info = tarea.datos()
    print(info.resumen())

    modelo = tarea.modelo(info).to(dispositivo)
    if cfg.entrenamiento.canales_last:
        modelo = modelo.to(memory_format=torch.channels_last)
    criterio = tarea.criterio(info, dispositivo)
    optimizador = crear_optimizador(modelo, cfg)
    usar_amp = dispositivo.type == "cuda" and cfg.entrenamiento.precision != "fp32"
    tipo_amp = torch.bfloat16 if cfg.entrenamiento.precision == "bf16" else torch.float16
    es_sam = isinstance(optimizador, SAM)

    print(f"\nMidiendo {args.lotes} lotes de {cfg.datos.batch} en {dispositivo.type}"
          f" ({cfg.entrenamiento.precision})…\n")

    # --- 1. entrenamiento completo (datos + GPU) ---
    modelo.train()
    segundos, muestras = _cronometrar(
        loader_train, args, dispositivo,
        lambda lote: _paso_completo(tarea, modelo, lote, criterio, optimizador,
                                    dispositivo, usar_amp, tipo_amp, es_sam))
    por_segundo = muestras / segundos

    # --- 2. solo datos, para ver quién manda ---
    segundos_datos, muestras_datos = _cronometrar(loader_train, args, dispositivo, None)
    datos_por_segundo = muestras_datos / segundos_datos

    # --- 3. validación ---
    modelo.eval()
    with torch.no_grad():
        segundos_val, muestras_val = _cronometrar(
            loader_val, args, dispositivo,
            lambda lote: tarea.paso(modelo, lote, criterio, dispositivo, entrenando=False))
    val_por_segundo = muestras_val / segundos_val

    # --- informe ---
    epocas = cfg.entrenamiento.epocas
    seg_epoca = info.n_train / por_segundo + info.n_val / val_por_segundo
    total = seg_epoca * epocas

    print(f"  entrenamiento : {por_segundo:>8.1f} muestras/s")
    print(f"  validación    : {val_por_segundo:>8.1f} muestras/s")
    print(f"  solo datos    : {datos_por_segundo:>8.1f} muestras/s  "
          f"(lo que la CPU puede servir)")

    if torch.cuda.is_available():
        print(f"  VRAM pico     : {torch.cuda.max_memory_allocated() / 1024**3:>8.2f} GB "
              f"de {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print(f"\n  por época     : {_tiempo(seg_epoca)}")
    print(f"  {epocas} épocas    : {_tiempo(total)}")
    if cfg.entrenamiento.paciencia:
        print(f"  (con parada temprana suele quedarse en la mitad o dos tercios)")

    _consejos(cfg, por_segundo, datos_por_segundo, dispositivo)


def _paso_completo(tarea, modelo, lote, criterio, optimizador, dispositivo,
                   usar_amp, tipo_amp, es_sam):
    optimizador.zero_grad(set_to_none=True)
    with torch.autocast(dispositivo.type, dtype=tipo_amp, enabled=usar_amp):
        paso = tarea.paso(modelo, lote, criterio, dispositivo, entrenando=True)
    paso.perdida.backward()
    if es_sam:
        optimizador.primer_paso()
        optimizador.zero_grad(set_to_none=True)
        with torch.autocast(dispositivo.type, dtype=tipo_amp, enabled=usar_amp):
            paso2 = tarea.paso(modelo, lote, criterio, dispositivo, entrenando=True)
        paso2.perdida.backward()
        optimizador.segundo_paso()
    else:
        optimizador.step()
    return paso


def _cronometrar(loader, args, dispositivo, funcion):
    """Devuelve (segundos, muestras) tras descartar los lotes de calentamiento."""
    iterador = iter(loader)
    muestras = 0
    inicio = None

    for indice in range(args.calentamiento + args.lotes):
        try:
            lote = next(iterador)
        except StopIteration:
            iterador = iter(loader)
            lote = next(iterador)
        if funcion is not None:
            funcion(lote)
        if indice == args.calentamiento - 1:
            if dispositivo.type == "cuda":
                torch.cuda.synchronize()
            inicio = time.perf_counter()
        elif indice >= args.calentamiento:
            muestras += _tam_lote(lote)

    if dispositivo.type == "cuda":
        torch.cuda.synchronize()
    return max(1e-6, time.perf_counter() - inicio), max(1, muestras)


def _tam_lote(lote) -> int:
    primero = lote[0]
    if isinstance(primero, torch.Tensor):
        return primero.size(0)
    if isinstance(primero, (list, tuple)):
        return len(primero)
    if hasattr(primero, "keys"):          # lote de texto ya tokenizado
        return len(next(iter(primero.values())))
    return 1


def _tiempo(segundos: float) -> str:
    if segundos < 90:
        return f"{segundos:.0f} s"
    if segundos < 5400:
        return f"{segundos / 60:.1f} min"
    return f"{segundos / 3600:.1f} h"


def _consejos(cfg, por_segundo: float, datos_por_segundo: float, dispositivo) -> None:
    print()
    if dispositivo.type != "cuda":
        print("  · Estás midiendo en CPU: en la GPU será entre 20 y 60 veces más rápido.")
        return

    if datos_por_segundo < por_segundo * 1.15:
        print("  · CUELLO DE BOTELLA EN LA CPU: la GPU va más rápido de lo que el DataLoader "
              "sirve.\n    Prueba a subir datos.workers, activar datos.cache_ram=true o "
              "usar una política\n    de aumentos más barata (basica en vez de randaugment).")
    else:
        print("  · La GPU es el límite, que es lo correcto.")

    if cfg.optimizador.sam:
        print("  · SAM está activo: cuesta el doble por lote. Quítalo si tienes prisa.")
    if cfg.entrenamiento.resolucion_progresiva:
        print("  · Con resolución progresiva las primeras épocas son más rápidas que esta "
              "medida.")
    if cfg.entrenamiento.precision == "fp32":
        print("  · fp32: pasar a bf16 suele dar entre 1,5x y 2x sin perder precisión.")


if __name__ == "__main__":
    main()

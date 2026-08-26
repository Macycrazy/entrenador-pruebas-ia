#!/usr/bin/env python3
"""Exporta un modelo entrenado y mide su latencia.

    python exportar.py experimentos/calidad --formato onnx
    python exportar.py experimentos/calidad --formato int8      # cuantización dinámica
    python exportar.py experimentos/calidad --benchmark

Formatos: onnx (portátil, sirve con onnxruntime), torchscript (sin Python del proyecto),
int8 (más pequeño y rápido en CPU) y fp16 (para GPU).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

import torch

from nucleo.carga import cargar_modelo, metadatos


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exporta y mide un modelo entrenado")
    p.add_argument("experimento", type=Path, help="Carpeta del experimento o archivo .pt")
    p.add_argument("--formato", default="onnx",
                   choices=["onnx", "torchscript", "int8", "fp16", "todos"])
    p.add_argument("--salida", type=Path, help="Carpeta destino (por defecto, la del experimento)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--benchmark", action="store_true", help="Medir latencia por imagen")
    p.add_argument("--repeticiones", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = argumentos()
    modelo, ckpt = cargar_modelo(args.experimento)
    meta = metadatos(ckpt)
    tam = meta["tam_img"]
    ejemplo = torch.randn(1, 3, tam, tam)
    destino = args.salida or (args.experimento if args.experimento.is_dir()
                              else args.experimento.parent)
    destino.mkdir(parents=True, exist_ok=True)
    formatos = ["onnx", "torchscript", "int8", "fp16"] if args.formato == "todos" \
        else [args.formato]

    for formato in formatos:
        try:
            ruta = _exportar(formato, modelo, ejemplo, destino, args, meta)
        except ModuleNotFoundError as error:
            faltante = str(error).split("'")[1] if "'" in str(error) else str(error)
            print(f"{formato:<12} falta una dependencia: pip install {faltante}")
            continue
        except Exception as error:  # noqa: BLE001 - un formato roto no debe tumbar el resto
            print(f"{formato:<12} ERROR: {error}")
            continue
        if ruta:
            # El exportador de ONNX saca los pesos a un .data aparte: hay que contarlo
            # y avisar, porque copiar solo el .onnx deja el modelo sin pesos.
            sidecar = ruta.with_suffix(ruta.suffix + ".data")
            total = ruta.stat().st_size + (sidecar.stat().st_size if sidecar.exists() else 0)
            print(f"{formato:<12} {ruta}  ({total / 1e6:.1f} MB)")
            if sidecar.exists():
                print(f"{'':<12} + {sidecar.name} — los dos archivos viajan juntos")

    (destino / "modelo.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    print(f"metadatos    {destino / 'modelo.meta.json'}")

    if args.benchmark:
        medir(modelo, ejemplo, args.repeticiones)


def _exportar(formato: str, modelo, ejemplo, destino: Path, args, meta) -> Path | None:
    if formato == "onnx":
        ruta = destino / "modelo.onnx"
        torch.onnx.export(modelo, ejemplo, str(ruta), input_names=["entrada"],
                          output_names=["logits"], opset_version=args.opset,
                          dynamic_axes={"entrada": {0: "lote"}, "logits": {0: "lote"}})
        return ruta

    if formato == "torchscript":
        ruta = destino / "modelo.torchscript.pt"
        torch.jit.trace(modelo, ejemplo).save(str(ruta))
        return ruta

    if formato == "int8":
        # Cuantización dinámica: solo las capas lineales, sin datos de calibración.
        # En CPU suele dar 2-3x de velocidad y 4x menos tamaño.
        cuantizado = torch.quantization.quantize_dynamic(
            modelo.cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8)
        ruta = destino / "modelo_int8.pt"
        torch.save({"state_dict": cuantizado.state_dict(), **meta}, ruta)
        return ruta

    if formato == "fp16":
        ruta = destino / "modelo_fp16.pt"
        torch.save({"state_dict": modelo.half().state_dict(), **meta}, ruta)
        modelo.float()
        return ruta
    return None


@torch.no_grad()
def medir(modelo, ejemplo, repeticiones: int) -> None:
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = modelo.to(dispositivo).eval()
    print(f"\nLatencia en {dispositivo}:")
    for lote in (1, 8, 32):
        x = ejemplo.repeat(lote, 1, 1, 1).to(dispositivo)
        for _ in range(5):
            modelo(x)
        if dispositivo == "cuda":
            torch.cuda.synchronize()
        inicio = time.perf_counter()
        for _ in range(repeticiones):
            modelo(x)
        if dispositivo == "cuda":
            torch.cuda.synchronize()
        total = (time.perf_counter() - inicio) / repeticiones
        print(f"  lote {lote:>3}: {total * 1000:>7.1f} ms  "
              f"({lote / total:>7.0f} img/s · {total / lote * 1000:.2f} ms por imagen)")


if __name__ == "__main__":
    main()

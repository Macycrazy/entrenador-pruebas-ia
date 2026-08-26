"""Comprueba que PyTorch ve la RTX 5060 Ti y puede usarla.

La serie RTX 50 (Blackwell) es sm_120: necesita PyTorch >= 2.7 compilado con CUDA 12.8.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
"""

from __future__ import annotations

import time

import torch


def main() -> None:
    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA (torch): {torch.version.cuda}")
    print(f"Arquitecturas compiladas: {', '.join(torch.cuda.get_arch_list()) or 'ninguna'}")

    if not torch.cuda.is_available():
        print("\nNo se detecta GPU. Revisa el driver NVIDIA y que torch sea la build cu128.")
        return

    props = torch.cuda.get_device_properties(0)
    sm = f"sm_{props.major}{props.minor}"
    print(f"\nGPU         : {props.name}")
    print(f"VRAM        : {props.total_memory / 1024**3:.1f} GB")
    print(f"Capacidad   : {sm}")
    print(f"bfloat16    : {'sí' if torch.cuda.is_bf16_supported() else 'no'}")

    if sm not in torch.cuda.get_arch_list():
        print(f"\nAVISO: esta build de PyTorch no incluye kernels para {sm}. "
              "Reinstala con el índice cu128 (ver cabecera de este archivo).")
        return

    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ a
    torch.cuda.synchronize()
    inicio = time.perf_counter()
    for _ in range(10):
        a @ a
    torch.cuda.synchronize()
    seg = (time.perf_counter() - inicio) / 10
    print(f"\nMatmul 8192³ bf16: {seg * 1000:.1f} ms  ({2 * 8192**3 / seg / 1e12:.1f} TFLOPS)")
    print("Todo listo para entrenar.")


if __name__ == "__main__":
    main()

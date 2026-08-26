#!/usr/bin/env python3
"""Usa un modelo de lenguaje ajustado con LoRA.

    python chatear.py experimentos/llm --pregunta "Clasifica: el carnet no sirve"
    python chatear.py experimentos/llm                 # modo interactivo

Carga el modelo base de su repositorio y le aplica encima el adaptador entrenado
(unos pocos MB), que es justo lo que guarda el entrenamiento.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

import torch

from nucleo.config import Config


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera texto con un LLM ajustado")
    p.add_argument("experimento", type=Path, help="Carpeta del experimento o .pt")
    p.add_argument("--pregunta", help="Instrucción; sin ella entra en modo interactivo")
    p.add_argument("--entrada", default="", help="Texto de contexto para la instrucción")
    p.add_argument("--tokens", type=int, default=200, help="Máximo de tokens a generar")
    p.add_argument("--temperatura", type=float, default=0.7)
    p.add_argument("--sin-adaptador", action="store_true",
                   help="Cargar solo el modelo base, para comparar con y sin ajuste")
    return p.parse_args()


def cargar(ruta: Path, sin_adaptador: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    archivo = ruta / "mejor.pt" if ruta.is_dir() else ruta
    ckpt = torch.load(archivo, map_location="cpu", weights_only=False)
    cfg = Config(ckpt["config"])
    base = ckpt.get("modelo_base", cfg.llm.modelo_base)

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizador = AutoTokenizer.from_pretrained(base)
    if tokenizador.pad_token is None:
        tokenizador.pad_token = tokenizador.eos_token

    modelo = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16 if dispositivo == "cuda" else torch.float32)

    if not sin_adaptador:
        from peft import LoraConfig, get_peft_model
        configuracion = LoraConfig(
            r=cfg.llm.lora_r, lora_alpha=cfg.llm.lora_alpha, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM",
            target_modules=cfg.llm.lora_modulos or "all-linear")
        modelo = get_peft_model(modelo, configuracion)
        faltan, sobran = modelo.load_state_dict(ckpt["state_dict"], strict=False)
        cargados = len(ckpt["state_dict"])
        if sobran:
            print(f"AVISO: {len(sobran)} pesos del checkpoint no encajan en el modelo.")
        print(f"Adaptador LoRA aplicado ({cargados} tensores)")

    modelo.eval().to(dispositivo)
    return modelo, tokenizador, cfg, dispositivo


@torch.no_grad()
def responder(modelo, tokenizador, cfg, dispositivo, instruccion: str, entrada: str,
              tokens: int, temperatura: float) -> str:
    prompt = cfg.llm.plantilla.format(instruccion=instruccion, entrada=entrada).rstrip() + "\n"
    entradas = tokenizador(prompt, return_tensors="pt").to(dispositivo)
    salida = modelo.generate(
        **entradas, max_new_tokens=tokens, do_sample=temperatura > 0,
        temperature=max(0.01, temperatura), top_p=0.9,
        pad_token_id=tokenizador.pad_token_id)
    generado = salida[0][entradas["input_ids"].shape[1]:]
    return tokenizador.decode(generado, skip_special_tokens=True).strip()


def main() -> None:
    args = argumentos()
    modelo, tokenizador, cfg, dispositivo = cargar(args.experimento, args.sin_adaptador)

    if args.pregunta:
        print("\n" + responder(modelo, tokenizador, cfg, dispositivo, args.pregunta,
                               args.entrada, args.tokens, args.temperatura))
        return

    print("\nModo interactivo. Ctrl-C o línea vacía para salir.\n")
    try:
        while True:
            instruccion = input("› ").strip()
            if not instruccion:
                break
            print(responder(modelo, tokenizador, cfg, dispositivo, instruccion, "",
                            args.tokens, args.temperatura) + "\n")
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()

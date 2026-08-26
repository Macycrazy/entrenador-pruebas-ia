"""Tarea: ajuste fino de un modelo de lenguaje con LoRA (o QLoRA en 4 bits).

En lugar de mover los miles de millones de parámetros del modelo, LoRA entrena unas
matrices pequeñas añadidas a cada capa de atención: entre el 0,1 % y el 1 % del total.
Por eso cabe en 16 GB lo que de otro modo necesitaría varios cientos.

    pip install transformers peft
    pip install bitsandbytes      # solo si usas llm.cuatro_bits (QLoRA)

La métrica es la **perplejidad**: cuánto «se sorprende» el modelo ante el texto correcto.
Más baja es mejor; 1,0 sería adivinar siempre la palabra exacta.
"""

from __future__ import annotations

import math

import torch

from nucleo.tarea import InfoDatos, Paso, Tarea, registrar

from . import datos as datos_mod


@registrar("texto_llm")
class TareaTextoLLM(Tarea):

    def datos(self):
        self.tokenizador = _tokenizador(self.cfg.llm.modelo_base)
        train, val = datos_mod.recopilar(self.cfg)
        loader_train, loader_val = datos_mod.crear_loaders(
            self.cfg, train, val, self.tokenizador)
        print(f"LLM: {self.cfg.llm.modelo_base} · hasta {self.cfg.llm.longitud_max} tokens · "
              f"{'solo la respuesta' if self.cfg.llm.solo_respuesta else 'todo el texto'} "
              f"cuenta para la pérdida")
        return loader_train, loader_val, InfoDatos(
            clases=["texto"], conteo=[len(train)], n_train=len(train), n_val=len(val))

    def modelo(self, info: InfoDatos):
        from transformers import AutoModelForCausalLM
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise SystemExit("El ajuste con LoRA necesita:  pip install peft") from None

        llm = self.cfg.llm
        extra = {}
        if llm.cuatro_bits:
            extra = {"quantization_config": _config_4bits()}

        modelo = AutoModelForCausalLM.from_pretrained(
            llm.modelo_base, dtype=torch.bfloat16, **extra)
        modelo.config.use_cache = False          # incompatible con gradient checkpointing
        if self.cfg.modelo.checkpoint_gradiente:
            modelo.gradient_checkpointing_enable()
            modelo.enable_input_require_grads()

        configuracion = LoraConfig(
            r=llm.lora_r, lora_alpha=llm.lora_alpha, lora_dropout=llm.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=llm.lora_modulos or "all-linear",
        )
        modelo = get_peft_model(modelo, configuracion)

        entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
        total = sum(p.numel() for p in modelo.parameters())
        print(f"LoRA: {entrenables:,} parámetros entrenables de {total:,} "
              f"({100 * entrenables / total:.2f} %)".replace(",", "."))
        return modelo

    def criterio(self, info: InfoDatos, dispositivo):
        return None      # la pérdida la calcula el propio modelo con las etiquetas

    def paso(self, modelo, lote, criterio, dispositivo, entrenando: bool,
             espejo: bool = False) -> Paso:
        entradas, etiquetas, _, _ = lote
        entradas = {k: v.to(dispositivo) for k, v in entradas.items()}
        etiquetas = etiquetas.to(dispositivo)
        salida = modelo(**entradas, labels=etiquetas)
        # Se devuelve un tensor por muestra para que el bucle promedie bien
        cuantas = torch.zeros(etiquetas.size(0))
        return Paso(perdida=salida.loss, logits=cuantas, objetivos=cuantas)

    def evaluador(self, info: InfoDatos):
        return EvaluadorLM(self.cfg)

    def descripcion(self) -> str:
        return f"{self.cfg.llm.modelo_base} (LoRA r={self.cfg.llm.lora_r})"

    def exportar_extra(self) -> dict:
        return {"modelo_base": self.cfg.llm.modelo_base,
                "lora_r": self.cfg.llm.lora_r,
                "plantilla": self.cfg.llm.plantilla,
                "longitud_max": self.cfg.llm.longitud_max}


class EvaluadorLM:
    """Perplejidad = e^(pérdida media). Es la métrica estándar en modelos de lenguaje."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.metrica_objetivo = cfg.entrenamiento.metrica_objetivo
        self.reiniciar()

    def reiniciar(self) -> None:
        self._suma, self._n = 0.0, 0

    def actualizar(self, logits=None, objetivos=None, perdida=None, subgrupos=None,
                   datos_extra=None) -> None:
        if perdida is not None:
            self._suma += perdida
            self._n += 1

    def resumen(self, calibrar: bool = False, curvas: bool = True) -> dict:
        if not self._n:
            return {}
        perdida = self._suma / self._n
        perplejidad = math.exp(min(20, perdida))
        return {
            # 'acc' se usa para elegir el mejor checkpoint: se invierte la perplejidad
            # para que "más alto = mejor" siga valiendo en todo el núcleo.
            "acc": 1.0 / perplejidad, "acc_balanceada": 1.0 / perplejidad,
            "perplejidad": perplejidad, "perdida": perdida, "n": self._n,
            "texto": f"perplejidad {perplejidad:.3f} (pérdida {perdida:.4f})",
        }


def _tokenizador(nombre: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit("Necesita:  pip install transformers") from None
    tok = AutoTokenizer.from_pretrained(nombre)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _config_4bits():
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise SystemExit("Necesita:  pip install transformers") from None
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

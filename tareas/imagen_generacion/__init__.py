"""Tarea: generación de imágenes con difusión + LoRA.

El modelo base ya sabe dibujar; lo que se entrena aquí es **enseñarle un sujeto nuevo**:
das 15-25 fotos de una persona u objeto y aprende a generarlo en situaciones que nunca
vio. Se entrena solo un adaptador LoRA sobre la U-Net (unos pocos MB), no el modelo entero.

    pip install diffusers transformers peft accelerate safetensors

Licencias: SD 1.5 y segmind/tiny-sd son OpenRAIL-M (uso libre con restricciones de uso);
SDXL es OpenRAIL++. **sdxl-turbo y sd-turbo son de uso NO comercial**: no los pongas en
producción sin leer su licencia.
"""

from __future__ import annotations

import torch

from nucleo.tarea import InfoDatos, Paso, Tarea, registrar

from . import datos as datos_mod


@registrar("imagen_generacion")
class TareaImagenGeneracion(Tarea):

    def datos(self):
        from transformers import CLIPTokenizer

        base = self.cfg.generacion.modelo_base
        self.tokenizador = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
        train, val = datos_mod.recopilar(self.cfg)
        loader_train, loader_val = datos_mod.crear_loaders(
            self.cfg, train, val, self.tokenizador)
        print(f"generación: {base} · {self.cfg.generacion.resolucion}px · "
              f"{len(train)} imágenes · instancia «{self.cfg.generacion.instancia}»")
        return loader_train, loader_val, InfoDatos(
            clases=["imagen"], conteo=[len(train)], n_train=len(train), n_val=len(val))

    def modelo(self, info: InfoDatos):
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel

        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise SystemExit("Necesita:  pip install peft") from None

        base = self.cfg.generacion.modelo_base
        self.vae = AutoencoderKL.from_pretrained(base, subfolder="vae").eval()
        self.codificador = CLIPTextModel.from_pretrained(base, subfolder="text_encoder").eval()
        self.planificador = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet")

        # VAE y codificador de texto se quedan congelados: solo se entrena la U-Net
        for parte in (self.vae, self.codificador):
            for p in parte.parameters():
                p.requires_grad_(False)

        g = self.cfg.generacion
        unet = get_peft_model(unet, LoraConfig(
            r=g.lora_r, lora_alpha=g.lora_alpha, init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
        entrenables = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        print(f"LoRA sobre la U-Net: {entrenables:,} parámetros entrenables".replace(",", "."))
        return unet

    def criterio(self, info: InfoDatos, dispositivo):
        self._dispositivo = dispositivo
        self.vae.to(dispositivo)
        self.codificador.to(dispositivo)
        return torch.nn.MSELoss()

    def paso(self, modelo, lote, criterio, dispositivo, entrenando: bool,
             espejo: bool = False) -> Paso:
        pixeles, ids, _, _ = lote
        pixeles = pixeles.to(dispositivo)
        ids = ids.to(dispositivo)

        with torch.no_grad():
            # La difusión no trabaja sobre píxeles sino sobre el espacio latente del VAE
            latentes = self.vae.encode(pixeles).latent_dist.sample()
            latentes = latentes * self.vae.config.scaling_factor
            contexto = self.codificador(ids)[0]

        ruido = torch.randn_like(latentes)
        pasos = torch.randint(0, self.planificador.config.num_train_timesteps,
                              (latentes.size(0),), device=dispositivo).long()
        sucias = self.planificador.add_noise(latentes, ruido, pasos)

        prediccion = modelo(sucias, pasos, encoder_hidden_states=contexto).sample
        objetivo = ruido if self.planificador.config.prediction_type == "epsilon" \
            else self.planificador.get_velocity(latentes, ruido, pasos)

        cuantas = torch.zeros(latentes.size(0))
        return Paso(perdida=criterio(prediccion.float(), objetivo.float()),
                    logits=cuantas, objetivos=cuantas)

    def evaluador(self, info: InfoDatos):
        return EvaluadorGeneracion()

    def descripcion(self) -> str:
        return f"{self.cfg.generacion.modelo_base} (LoRA)"

    def exportar_extra(self) -> dict:
        return {"modelo_base": self.cfg.generacion.modelo_base,
                "instancia": self.cfg.generacion.instancia,
                "resolucion": self.cfg.generacion.resolucion,
                "arquitectura": "difusion_lora"}


class EvaluadorGeneracion:
    """No hay métrica objetiva de «qué bonito quedó»: se sigue la pérdida de difusión.

    Lo que de verdad se mira es la galería de la vista: generar y juzgar con el ojo.
    """

    def __init__(self):
        self.metrica_objetivo = "acc"
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
        return {"acc": 1.0 / (1.0 + perdida), "acc_balanceada": 1.0 / (1.0 + perdida),
                "perdida": perdida, "n": self._n,
                "texto": f"pérdida de difusión {perdida:.4f} "
                         f"(baja despacio y sin picos = va bien)"}

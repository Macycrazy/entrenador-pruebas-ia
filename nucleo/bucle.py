"""Bucle de entrenamiento genérico: sirve para cualquier tarea del registro.

Reúne todo lo que no depende del tipo de dato: precisión mixta, acumulación de
gradiente, EMA/SWA, congelado progresivo, early stopping, reanudación y registro.
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn as nn

from nucleo import metricas as metricas_mod
from nucleo.ema import EMA, SWA
from nucleo.optimizadores import SAM, crear_optimizador, crear_scheduler
from nucleo.vigilante import SobrecalentamientoGPU, VigilanteGPU

NOMBRES_CABEZA = ("fc", "classifier", "head", "heads", "cabeza", "cabezas_extra", "cuello",
              "roi_heads", "rpn")


class Entrenador:
    def __init__(self, cfg, tarea, experimento):
        self.cfg, self.tarea, self.exp = cfg, tarea, experimento
        self.dispositivo = preparar_dispositivo(cfg)

    # ------------------------------------------------------------------ montaje

    def preparar(self):
        e = self.cfg.entrenamiento
        self.loader_train, self.loader_val, self.info = self.tarea.datos()
        print(self.info.resumen())

        self.modelo = self.tarea.modelo(self.info).to(self.dispositivo)
        if e.canales_last:
            self.modelo = self.modelo.to(memory_format=torch.channels_last)
        if e.compilar:
            self.modelo = torch.compile(self.modelo)

        self.criterio = self.tarea.criterio(self.info, self.dispositivo)
        self.optimizador = crear_optimizador(self.modelo, self.cfg)
        pasos = max(1, len(self.loader_train) // max(1, e.acumular))
        self.scheduler, self.sched_por_paso = crear_scheduler(self.optimizador, self.cfg, pasos)

        self.usar_amp = self.dispositivo.type == "cuda" and e.precision != "fp32"
        self.tipo_amp = torch.bfloat16 if e.precision == "bf16" else torch.float16
        self.escalador = torch.amp.GradScaler(
            "cuda", enabled=self.usar_amp and e.precision == "fp16")

        self.ema = EMA(self.modelo, e.ema) if e.ema > 0 else None
        self.swa = SWA(self.modelo) if e.swa else None
        self.evaluador = self.tarea.evaluador(self.info)
        self.vigilante = VigilanteGPU(e.temp_max, e.temp_avisos,
                                      activo=e.vigilar_gpu and self.dispositivo.type == "cuda")

        self.epoca_inicial = 0
        if e.reanudar:
            self.exp.reanudar_desde = e.reanudar
            self.epoca_inicial = self.exp.cargar_para_reanudar(
                self.modelo, self.optimizador, self.scheduler, self.ema)
        return self

    # ------------------------------------------------------------------ ejecución

    def ejecutar(self) -> dict:
        e = self.cfg.entrenamiento
        mejor, mejor_epoca, mejor_resumen = -float("inf"), -1, {}
        print(f"\n{self.cfg.nombre} · {e.epocas} épocas · {self.tarea.descripcion()} · "
              f"{e.precision} · lote {self.cfg.datos.batch}×{e.acumular} · "
              f"{self.cfg.optimizador.nombre}/{self.cfg.scheduler.nombre}\n")

        for epoca in range(self.epoca_inicial, e.epocas):
            self._ajustar_congelado(epoca)
            self.tarea.al_cambiar_epoca(epoca, self.modelo)
            inicio = time.perf_counter()

            try:
                perdida_train = self._epoca_entrenamiento(epoca)
            except SobrecalentamientoGPU as error:
                # Parada limpia: el estado ya está en ultimo.pt de la época anterior,
                # pero se vuelve a guardar por si acaso.
                self.exp.guardar("ultimo.pt", self.modelo, {"parada": str(error)},
                                 self.optimizador, self.scheduler, self.ema, self.swa,
                                 max(0, epoca - 1))
                print(f"\nENTRENAMIENTO DETENIDO: {error}\n"
                      f"Revisa ventilación y polvo. Para continuar cuando esté fresca:\n"
                      f"  python entrenar.py --reanudar {self.exp.dir}")
                break
            resumen = self._validar()

            if self.swa and epoca >= e.swa_desde * e.epocas:
                self.swa.actualizar(self.modelo)

            valor = resumen.get(e.metrica_objetivo, resumen.get("acc", 0.0))
            if not self.sched_por_paso:
                self.scheduler.step(valor) if self.cfg.scheduler.nombre == "plateau" \
                    else self.scheduler.step()

            segundos = time.perf_counter() - inicio
            lr = self.optimizador.param_groups[-1]["lr"]
            marca = ""
            if valor > mejor:
                mejor, mejor_epoca, mejor_resumen = valor, epoca, resumen
                self.exp.guardar("mejor.pt", self._modelo_para_guardar(),
                                 self._extra(resumen), epoca=epoca)
                marca = "  <- mejor"

            self.exp.guardar("ultimo.pt", self.modelo, self._extra(resumen),
                             self.optimizador, self.scheduler, self.ema, self.swa, epoca)
            if self.cfg.salida.guardar_cada and (epoca + 1) % self.cfg.salida.guardar_cada == 0:
                self.exp.guardar(f"epoca_{epoca + 1:03d}.pt", self.modelo,
                                 self._extra(resumen), epoca=epoca)

            detalle = " · ".join(f"{c} {m['recall']:.3f}"
                                 for c, m in resumen.get("por_clase", {}).items())
            print(f"época {epoca + 1:>3}/{e.epocas} | train {perdida_train:.4f} | "
                  f"val {resumen.get('perdida', 0):.4f}/{resumen.get('acc', 0):.4f} | "
                  f"{detalle} | lr {lr:.2e} | {segundos:.0f}s{marca}")

            self.exp.fila({"epoca": epoca + 1, **self.vigilante.resumen(),
                           "perdida_train": round(perdida_train, 5),
                           "perdida_val": round(resumen.get("perdida", 0), 5),
                           "acc": round(resumen.get("acc", 0), 5),
                           "acc_balanceada": round(resumen.get("acc_balanceada", 0), 5),
                           "auc": round(resumen.get("auc", float("nan")), 5),
                           "ece": round(resumen.get("ece", 0), 5), "lr": lr,
                           "segundos": round(segundos, 1)})

            if e.paciencia and epoca - mejor_epoca >= e.paciencia:
                print(f"\nSin mejora en {e.paciencia} épocas: parada temprana.")
                break

        if self.swa and self.swa.n:
            print("\nRecalculando BatchNorm para el promedio SWA…")
            self.swa.recalcular_bn(self.loader_train, self.dispositivo)
            resumen_swa = self._validar(modelo=self.swa.media)
            print("SWA → " + metricas_mod.formatear(resumen_swa, self.info.clases).splitlines()[0])
            if resumen_swa.get(e.metrica_objetivo, 0) > mejor:
                mejor, mejor_resumen = resumen_swa[e.metrica_objetivo], resumen_swa
                self.exp.guardar("mejor.pt", self.swa.media, self._extra(resumen_swa))
                print("El promedio SWA mejora: guardado como mejor.pt")

        print(f"\nMejor {e.metrica_objetivo}: {mejor:.4f} (época {mejor_epoca + 1})")
        print(metricas_mod.formatear(mejor_resumen, self.info.clases))
        self.exp.informe({"mejor": mejor_resumen, "epoca": mejor_epoca + 1,
                          "config": dict(self.cfg)})
        self.exp.cerrar()
        return mejor_resumen

    # ------------------------------------------------------------------ una época

    def _epoca_entrenamiento(self, epoca: int) -> float:
        e = self.cfg.entrenamiento
        self.modelo.train()
        total, n = 0.0, 0
        inicio = time.perf_counter()
        self.vigilante.reiniciar()
        self.optimizador.zero_grad(set_to_none=True)
        es_sam = isinstance(self.optimizador, SAM)

        for i, lote in enumerate(self.loader_train):
            with torch.autocast(self.dispositivo.type, dtype=self.tipo_amp, enabled=self.usar_amp):
                paso = self.tarea.paso(self.modelo, lote, self.criterio,
                                       self.dispositivo, entrenando=True)
                perdida = paso.perdida / e.acumular

            self.escalador.scale(perdida).backward() if self.escalador.is_enabled() \
                else perdida.backward()

            if (i + 1) % e.acumular == 0:
                if es_sam:
                    self._paso_sam(lote)
                else:
                    self._paso_optimizador()
                if self.ema:
                    self.ema.actualizar(self.modelo)
                if self.sched_por_paso:
                    self.scheduler.step()

            total += paso.perdida.item() * paso.objetivos.size(0)
            n += paso.objetivos.size(0)
            lectura = self.vigilante.comprobar()

            if i % 20 == 0:
                ips = n / max(1e-6, time.perf_counter() - inicio)
                clima = f" | {lectura['temp']:.0f} °C {lectura['potencia']:.0f} W" \
                    if lectura else ""
                print(f"  lote {i:>5}/{len(self.loader_train)} | pérdida {paso.perdida.item():.4f}"
                      f" | {ips:.0f} muestras/s{clima}", end="\r", flush=True)

        print(" " * 96, end="\r")
        return total / max(1, n)

    def _paso_optimizador(self) -> None:
        clip = self.cfg.entrenamiento.clip_grad
        if self.escalador.is_enabled():
            self.escalador.unscale_(self.optimizador)
        if clip:
            nn.utils.clip_grad_norm_(self.modelo.parameters(), clip)
        if self.escalador.is_enabled():
            self.escalador.step(self.optimizador)
            self.escalador.update()
        else:
            self.optimizador.step()
        self.optimizador.zero_grad(set_to_none=True)

    def _paso_sam(self, lote) -> None:
        """SAM: sube a la cresta, vuelve a medir el gradiente y da el paso desde allí."""
        clip = self.cfg.entrenamiento.clip_grad
        if clip:
            nn.utils.clip_grad_norm_(self.modelo.parameters(), clip)
        self.optimizador.primer_paso()
        self.optimizador.zero_grad(set_to_none=True)
        with torch.autocast(self.dispositivo.type, dtype=self.tipo_amp, enabled=self.usar_amp):
            paso = self.tarea.paso(self.modelo, lote, self.criterio,
                                   self.dispositivo, entrenando=True)
        paso.perdida.backward()
        if clip:
            nn.utils.clip_grad_norm_(self.modelo.parameters(), clip)
        self.optimizador.segundo_paso()
        self.optimizador.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _validar(self, modelo=None) -> dict:
        modelo = modelo or self.modelo
        contexto = self.ema.aplicado(modelo) if self.ema else _nulo()
        with contexto:
            modelo.eval()
            self.evaluador.reiniciar()
            for lote in self.loader_val:
                with torch.autocast(self.dispositivo.type, dtype=self.tipo_amp,
                                    enabled=self.usar_amp):
                    paso = self.tarea.paso(modelo, lote, self.criterio,
                                           self.dispositivo, entrenando=False)
                logits = paso.logits
                if self.cfg.evaluacion.tta:
                    with torch.autocast(self.dispositivo.type, dtype=self.tipo_amp,
                                        enabled=self.usar_amp):
                        espejo = self.tarea.paso(modelo, lote, self.criterio, self.dispositivo,
                                                 entrenando=False, espejo=True)
                    logits = (logits.softmax(1) + espejo.logits.softmax(1)).log()
                self.evaluador.actualizar(logits, paso.objetivos, paso.perdida.item(),
                                          paso.subgrupos, paso.datos_extra)
        return self.evaluador.resumen(calibrar=self.cfg.evaluacion.calibrar,
                                      curvas=self.cfg.evaluacion.curvas)

    # ------------------------------------------------------------------ auxiliares

    def _ajustar_congelado(self, epoca: int) -> None:
        e = self.cfg.entrenamiento
        if self.cfg.modelo.congelar_backbone:
            congelar(self.modelo, True)
            return
        if e.descongelado_gradual:
            fraccion = (epoca + 1) / max(1, e.epocas)
            descongelar_progresivo(self.modelo, fraccion)
        elif e.congelar_epocas:
            congelar(self.modelo, epoca < e.congelar_epocas)

    def _modelo_para_guardar(self):
        return self.ema.sombra if self.ema else self.modelo

    def _extra(self, resumen: dict) -> dict:
        return {"clases": self.info.clases, "metricas": resumen,
                "tarea": self.cfg.tarea, **self.tarea.exportar_extra()}


# ---------------------------------------------------------------- utilidades

class _nulo:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def preparar_dispositivo(cfg) -> torch.device:
    torch.manual_seed(cfg.semilla)
    random.seed(cfg.semilla)
    np.random.seed(cfg.semilla)
    if cfg.determinista:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    if not torch.cuda.is_available():
        print("AVISO: sin GPU, se entrenará en CPU (mucho más lento).")
        return torch.device("cpu")

    if cfg.entrenamiento.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} · {props.total_memory / 1024**3:.1f} GB · "
          f"sm_{props.major}{props.minor} · torch {torch.__version__}")
    return torch.device("cuda")


def congelar(modelo: nn.Module, valor: bool) -> None:
    """Congela todo salvo la cabeza clasificadora."""
    base = getattr(modelo, "_orig_mod", modelo)
    for nombre, parametro in base.named_parameters():
        es_cabeza = nombre.split(".")[0] in NOMBRES_CABEZA
        parametro.requires_grad_(es_cabeza or not valor)


def descongelar_progresivo(modelo: nn.Module, fraccion: float) -> None:
    """Va soltando bloques de atrás hacia delante conforme avanza el entrenamiento."""
    base = getattr(modelo, "_orig_mod", modelo)
    hijos = list(base.named_children())
    cuantos = max(1, int(round(fraccion * len(hijos))))
    descongelados = {n for n, _ in hijos[-cuantos:]} | set(NOMBRES_CABEZA)
    for nombre, parametro in base.named_parameters():
        parametro.requires_grad_(nombre.split(".")[0] in descongelados)

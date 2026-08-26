"""Optimizadores, planificadores de learning rate y buscador automático de lr."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim import lr_scheduler


# ---------------------------------------------------------------- grupos de parámetros

def grupos_parametros(modelo: nn.Module, cfg) -> list[dict]:
    """Separa backbone y cabeza para poder darles learning rates distintos.

    Además saca de weight decay los sesgos y las normalizaciones, que es lo estándar
    y evita penalizar parámetros que no lo necesitan.
    """
    cabeza_nombres = ("fc", "classifier", "head", "heads", "cabeza", "cabezas_extra", "cuello",
                      "roi_heads", "rpn")
    grupos = {"backbone_wd": [], "backbone_sin_wd": [], "cabeza_wd": [], "cabeza_sin_wd": []}

    # Se incluyen también los parámetros congelados: el bucle los descongela más tarde
    # y si no estuvieran en el optimizador nunca llegarían a entrenarse.
    for nombre, parametro in modelo.named_parameters():
        es_cabeza = nombre.split(".")[0] in cabeza_nombres
        sin_wd = parametro.ndim <= 1 or nombre.endswith(".bias")
        clave = ("cabeza" if es_cabeza else "backbone") + ("_sin_wd" if sin_wd else "_wd")
        grupos[clave].append(parametro)

    lr, wd, factor = cfg.optimizador.lr, cfg.optimizador.wd, cfg.optimizador.lr_backbone_factor
    salida = [
        {"params": grupos["backbone_wd"], "lr": lr * factor, "weight_decay": wd},
        {"params": grupos["backbone_sin_wd"], "lr": lr * factor, "weight_decay": 0.0},
        {"params": grupos["cabeza_wd"], "lr": lr, "weight_decay": wd},
        {"params": grupos["cabeza_sin_wd"], "lr": lr, "weight_decay": 0.0},
    ]
    return [g for g in salida if g["params"]]


# ---------------------------------------------------------------- optimizadores

def crear_optimizador(modelo: nn.Module, cfg):
    grupos = grupos_parametros(modelo, cfg)
    o = cfg.optimizador
    nombre = o.nombre.lower()
    comunes = {"lr": o.lr, "weight_decay": o.wd}

    if nombre == "sgd":
        opt = torch.optim.SGD(grupos, momentum=o.momentum, nesterov=o.nesterov, **comunes)
    elif nombre == "adam":
        opt = torch.optim.Adam(grupos, betas=tuple(o.betas), eps=o.eps, **comunes)
    elif nombre in ("adamw", "adamw_fused"):
        extra = {"fused": True} if nombre == "adamw_fused" and torch.cuda.is_available() else {}
        opt = torch.optim.AdamW(grupos, betas=tuple(o.betas), eps=o.eps, **comunes, **extra)
    elif nombre == "nadam":
        opt = torch.optim.NAdam(grupos, betas=tuple(o.betas), eps=o.eps, **comunes)
    elif nombre == "radam":
        opt = torch.optim.RAdam(grupos, betas=tuple(o.betas), eps=o.eps, **comunes)
    elif nombre == "adamax":
        opt = torch.optim.Adamax(grupos, betas=tuple(o.betas), eps=o.eps, **comunes)
    elif nombre == "rmsprop":
        opt = torch.optim.RMSprop(grupos, momentum=o.momentum, eps=o.eps, **comunes)
    elif nombre == "lion":
        opt = Lion(grupos, betas=tuple(o.betas[:2]), **comunes)
    elif nombre == "adam8bit":
        opt = _adam8bit(grupos, comunes, o)
    else:
        raise SystemExit(f"Optimizador '{o.nombre}' desconocido")

    if o.lookahead:
        opt = Lookahead(opt)
    if o.sam:
        opt = SAM(opt, rho=o.sam_rho)
    return opt


def _adam8bit(grupos, comunes, o):
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise SystemExit("adam8bit necesita bitsandbytes:  pip install bitsandbytes") from None
    return bnb.optim.AdamW8bit(grupos, betas=tuple(o.betas), eps=o.eps, **comunes)


class Lion(torch.optim.Optimizer):
    """EvoLved Sign Momentum. Menos memoria que AdamW y suele ir bien con lr ~3-10x menor."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        perdida = closure() if closure is not None else None
        for grupo in self.param_groups:
            beta1, beta2 = grupo["betas"]
            for p in grupo["params"]:
                if p.grad is None:
                    continue
                estado = self.state[p]
                if "momentum" not in estado:
                    estado["momentum"] = torch.zeros_like(p)
                m = estado["momentum"]
                p.mul_(1 - grupo["lr"] * grupo["weight_decay"])
                actualizacion = m.mul(beta1).add_(p.grad, alpha=1 - beta1).sign_()
                p.add_(actualizacion, alpha=-grupo["lr"])
                m.mul_(beta2).add_(p.grad, alpha=1 - beta2)
        return perdida


class Lookahead(torch.optim.Optimizer):
    """Envuelve otro optimizador y cada k pasos interpola hacia los pesos «lentos»."""

    def __init__(self, base, k: int = 5, alfa: float = 0.5):
        self.base, self.k, self.alfa, self.paso = base, k, alfa, 0
        self.param_groups = base.param_groups
        self.state = base.state
        self.defaults = base.defaults
        self._lentos = [[p.clone().detach() for p in g["params"]] for g in base.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        self.base.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        perdida = self.base.step(closure)
        self.paso += 1
        if self.paso % self.k == 0:
            for grupo, lentos in zip(self.param_groups, self._lentos):
                for rapido, lento in zip(grupo["params"], lentos):
                    lento.add_(rapido - lento, alpha=self.alfa)
                    rapido.copy_(lento)
        return perdida

    def state_dict(self):
        return self.base.state_dict()

    def load_state_dict(self, estado):
        self.base.load_state_dict(estado)


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization: dos pasadas por lote, busca mínimos planos.

    El bucle de entrenamiento lo detecta y hace forward/backward dos veces.
    """

    def __init__(self, base, rho: float = 0.05):
        self.base, self.rho = base, rho
        self.param_groups = base.param_groups
        self.state = base.state
        self.defaults = base.defaults

    def zero_grad(self, set_to_none: bool = True):
        self.base.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def primer_paso(self):
        norma = self._norma_gradiente()
        for grupo in self.param_groups:
            escala = self.rho / (norma + 1e-12)
            for p in grupo["params"]:
                if p.grad is None:
                    continue
                e = p.grad * escala
                p.add_(e)
                self.state[p]["e"] = e

    @torch.no_grad()
    def segundo_paso(self):
        for grupo in self.param_groups:
            for p in grupo["params"]:
                if "e" in self.state[p]:
                    p.sub_(self.state[p].pop("e"))
        self.base.step()
        # El scheduler vigila que se haya llamado a optimizador.step(); aquí el paso
        # real lo da `base`, así que hay que avisarle o suelta un warning por época.
        self._opt_called = True

    def step(self, closure=None):
        raise RuntimeError("SAM se usa con primer_paso()/segundo_paso() desde el bucle")

    def _norma_gradiente(self):
        return torch.norm(torch.stack([
            p.grad.norm(p=2) for g in self.param_groups for p in g["params"] if p.grad is not None
        ]), p=2)

    def state_dict(self):
        return self.base.state_dict()

    def load_state_dict(self, estado):
        self.base.load_state_dict(estado)


# ---------------------------------------------------------------- schedulers

def crear_scheduler(optimizador, cfg, pasos_por_epoca: int):
    """Devuelve (scheduler, por_paso). `por_paso` indica si avanza por lote o por época."""
    s = cfg.scheduler
    epocas = cfg.entrenamiento.epocas
    total = max(1, epocas * pasos_por_epoca)
    warmup = int(s.warmup_epocas * pasos_por_epoca)
    nombre = s.nombre.lower()

    if nombre == "constante":
        return lr_scheduler.LambdaLR(optimizador, lambda _: 1.0), True

    if nombre == "onecycle":
        maximos = [g["lr"] for g in optimizador.param_groups]
        return lr_scheduler.OneCycleLR(
            optimizador, max_lr=maximos, total_steps=total,
            pct_start=max(0.05, s.warmup_epocas / max(1, epocas)),
        ), True

    if nombre == "coseno":
        def factor(paso):
            if paso < warmup:
                return (paso + 1) / max(1, warmup)
            avance = (paso - warmup) / max(1, total - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, avance)))
        return lr_scheduler.LambdaLR(optimizador, factor), True

    if nombre == "polinomial":
        def factor(paso):
            if paso < warmup:
                return (paso + 1) / max(1, warmup)
            avance = (paso - warmup) / max(1, total - warmup)
            return max(0.0, (1 - min(1.0, avance)) ** s.potencia)
        return lr_scheduler.LambdaLR(optimizador, factor), True

    # --- los siguientes avanzan una vez por época ---
    if nombre == "coseno_reinicios":
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizador, T_0=s.T0, eta_min=s.min_lr), False
    if nombre == "step":
        return lr_scheduler.StepLR(optimizador, step_size=s.step_tam, gamma=s.gamma), False
    if nombre == "multistep":
        return lr_scheduler.MultiStepLR(optimizador, milestones=s.hitos, gamma=s.gamma), False
    if nombre == "exponencial":
        return lr_scheduler.ExponentialLR(optimizador, gamma=s.gamma), False
    if nombre == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizador, mode="max", factor=s.gamma, patience=s.paciencia_plateau), False

    raise SystemExit(f"Scheduler '{s.nombre}' desconocido")


# ---------------------------------------------------------------- LR finder

@torch.no_grad()
def _suavizar(valores: list[float], beta: float = 0.05) -> list[float]:
    salida, media = [], valores[0]
    for v in valores:
        media = (1 - beta) * media + beta * v
        salida.append(media)
    return salida


def buscar_lr(modelo, loader, criterio, dispositivo, cfg,
              lr_min: float = 1e-7, lr_max: float = 1.0, pasos: int = 100) -> dict:
    """Test de rango de learning rate (Smith): sube el lr y mira dónde explota la pérdida.

    Devuelve el lr sugerido (el del descenso más pronunciado, un orden por debajo del mínimo).
    """
    modelo.train()
    estado_inicial = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
    optimizador = crear_optimizador(modelo, cfg)
    if isinstance(optimizador, (SAM, Lookahead)):
        optimizador = optimizador.base

    gamma = (lr_max / lr_min) ** (1 / max(1, pasos - 1))
    lrs, perdidas = [], []
    lr = lr_min

    iterador = iter(loader)
    for paso in range(pasos):
        try:
            x, y = next(iterador)[:2]
        except StopIteration:
            iterador = iter(loader)
            x, y = next(iterador)[:2]
        x, y = x.to(dispositivo), y.to(dispositivo)

        for grupo in optimizador.param_groups:
            grupo["lr"] = lr
        optimizador.zero_grad(set_to_none=True)
        perdida = criterio(modelo(x), y)
        perdida.backward()
        optimizador.step()

        valor = perdida.item()
        if paso > 0 and (valor > 4 * min(perdidas) or math.isnan(valor)):
            break
        lrs.append(lr)
        perdidas.append(valor)
        lr *= gamma

    modelo.load_state_dict(estado_inicial)
    if len(perdidas) < 5:
        return {"sugerido": cfg.optimizador.lr, "lrs": lrs, "perdidas": perdidas}

    suaves = _suavizar(perdidas)
    pendientes = [suaves[i + 1] - suaves[i] for i in range(len(suaves) - 1)]
    indice = min(range(len(pendientes)), key=lambda i: pendientes[i])
    return {"sugerido": lrs[indice], "lrs": lrs, "perdidas": suaves}

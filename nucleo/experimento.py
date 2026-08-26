"""Carpeta de experimento: checkpoints, reanudación, historial y registro de métricas."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import torch

from nucleo import config as config_mod


class Experimento:
    """Todo lo que produce un entrenamiento vive junto:

        experimentos/<nombre>/
            config.yaml       la configuración exacta usada
            mejor.pt          checkpoint con la mejor métrica de validación
            ultimo.pt         último estado completo (para reanudar)
            historial.csv     una fila por época
            metricas.json     informe final
            errores.txt       ejemplos mal clasificados
    """

    def __init__(self, cfg, reanudar: str | Path | None = None):
        self.cfg = cfg
        self.dir = Path(cfg.salida.dir) / cfg.nombre
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv = self.dir / "historial.csv"
        self.escritor = self._abrir_tensorboard() if cfg.salida.tensorboard else None
        self.reanudar_desde = Path(reanudar) if reanudar else None
        config_mod.guardar(cfg, self.dir / "config.yaml")

    # ------------------------------------------------------------------ registro

    def fila(self, datos: dict) -> None:
        # Las columnas se fijan con la primera fila: si una época aporta claves nuevas
        # (o le falta alguna, como la temperatura antes de la primera lectura) se
        # rellenan o se descartan en vez de romper el CSV a mitad del entrenamiento.
        if not hasattr(self, "_columnas"):
            # Al reanudar, respetar las columnas del historial que ya existe.
            if self.csv.exists():
                with self.csv.open() as f:
                    cabecera = f.readline().strip()
                self._columnas = cabecera.split(",") if cabecera else list(datos)
            else:
                self._columnas = list(datos)
            nuevo = not self.csv.exists()
        else:
            nuevo = False
        fila = {clave: _plano(datos.get(clave, "")) for clave in self._columnas}
        with self.csv.open("a" if not nuevo else "w", newline="") as f:
            escritor = csv.DictWriter(f, fieldnames=self._columnas)
            if nuevo:
                escritor.writeheader()
            escritor.writerow(fila)
        if self.escritor:
            for clave, valor in datos.items():
                if isinstance(valor, (int, float)):
                    self.escritor.add_scalar(clave, valor, datos.get("epoca", 0))

    def informe(self, metricas: dict) -> None:
        (self.dir / "metricas.json").write_text(
            json.dumps(metricas, indent=2, ensure_ascii=False, default=str))

    def texto(self, nombre: str, contenido: str) -> None:
        (self.dir / nombre).write_text(contenido)

    def cerrar(self) -> None:
        if self.escritor:
            self.escritor.close()

    # ------------------------------------------------------------------ checkpoints

    def guardar(self, nombre: str, modelo, extra: dict, optimizador=None,
                scheduler=None, ema=None, swa=None, epoca: int = 0) -> Path:
        estado = {
            "state_dict": _pesos_a_guardar(modelo),
            "config": dict(self.cfg),
            "epoca": epoca,
            "fecha": datetime.now().isoformat(timespec="seconds"),
            **extra,
        }
        if optimizador is not None:
            estado["optimizador"] = optimizador.state_dict()
        if scheduler is not None:
            estado["scheduler"] = scheduler.state_dict()
        if ema is not None:
            estado["ema"] = ema.state_dict()
        if swa is not None:
            estado["swa"] = _sin_compilar(swa.media).state_dict()
        ruta = self.dir / nombre
        torch.save(estado, ruta)
        return ruta

    def cargar_para_reanudar(self, modelo, optimizador, scheduler, ema=None) -> int:
        """Restaura modelo, optimizador, scheduler y EMA. Devuelve la época siguiente."""
        ruta = self.reanudar_desde
        if ruta and ruta.is_dir():
            ruta = ruta / "ultimo.pt"
        if not ruta or not ruta.exists():
            raise SystemExit(f"No hay checkpoint para reanudar en {ruta}")

        estado = torch.load(ruta, map_location="cpu", weights_only=False)
        _sin_compilar(modelo).load_state_dict(estado["state_dict"])
        if optimizador is not None and "optimizador" in estado:
            optimizador.load_state_dict(estado["optimizador"])
        if scheduler is not None and "scheduler" in estado:
            scheduler.load_state_dict(estado["scheduler"])
        if ema is not None and "ema" in estado:
            ema.load_state_dict(estado["ema"])
        print(f"Reanudando desde {ruta} (época {estado['epoca'] + 1})")
        return estado["epoca"] + 1

    def _abrir_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("AVISO: tensorboard no instalado, se omite el registro.")
            return None
        return SummaryWriter(str(self.dir / "tb"))


def _plano(valor):
    return json.dumps(valor, ensure_ascii=False) if isinstance(valor, (dict, list)) else valor


def _sin_compilar(modelo):
    return getattr(modelo, "_orig_mod", modelo)


def _pesos_a_guardar(modelo) -> dict:
    """Con LoRA solo se guardan los adaptadores.

    Un modelo de lenguaje de 1,5 B pesa ~3 GB; sus adaptadores, unos pocos MB. Guardar
    el modelo entero en cada época llenaría el disco sin aportar nada: los pesos base
    no cambian y se vuelven a descargar de su repositorio al cargar.
    """
    base = _sin_compilar(modelo)
    estado = base.state_dict()
    if hasattr(base, "peft_config"):
        return {k: v for k, v in estado.items()
                if "lora_" in k or "modules_to_save" in k}
    return estado

"""Tarea: detección de anomalías visuales — encontrar lo raro sin haberlo visto nunca.

La gracia está en que **solo hace falta enseñarle lo normal**. En control de calidad o
vigilancia nadie tiene fotos de todos los defectos posibles, pero de lo correcto hay a
montones. El modelo aprende a reconstruir lo normal; cuando le llega algo que no encaja,
la reconstrucción falla y ese error delata la anomalía — y además dice **dónde** está.

    datos_anomalias/normal/*.jpg      solo ejemplos correctos
    datos_anomalias/anomalas/*.jpg    opcional, SOLO para medir (nunca se entrena con ellos)

    python entrenar.py --config configs/anomalias.yaml
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from nucleo.metricas import _auc
from nucleo.tarea import InfoDatos, Paso, Tarea, registrar

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Autocodificador(nn.Module):
    """Comprime la imagen a unos pocos números y la reconstruye desde ahí."""

    def __init__(self, canales: int = 32, cuello: int = 64, tam: int = 128):
        super().__init__()
        c = canales
        self.codificador = nn.Sequential(
            _bloque(3, c), _bloque(c, c * 2), _bloque(c * 2, c * 4), _bloque(c * 4, c * 4),
        )
        lado = tam // 16
        self.aplanar = nn.Sequential(nn.Flatten(), nn.Linear(c * 4 * lado * lado, cuello))
        self.desaplanar = nn.Sequential(
            nn.Linear(cuello, c * 4 * lado * lado), nn.Unflatten(1, (c * 4, lado, lado)))
        self.decodificador = nn.Sequential(
            _bloque_sube(c * 4, c * 4), _bloque_sube(c * 4, c * 2),
            _bloque_sube(c * 2, c), _bloque_sube(c, c),
            nn.Conv2d(c, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decodificador(self.desaplanar(self.aplanar(self.codificador(x))))


def _bloque(dentro, fuera):
    return nn.Sequential(nn.Conv2d(dentro, fuera, 4, stride=2, padding=1),
                         nn.BatchNorm2d(fuera), nn.LeakyReLU(0.2, inplace=True))


def _bloque_sube(dentro, fuera):
    return nn.Sequential(nn.ConvTranspose2d(dentro, fuera, 4, stride=2, padding=1),
                         nn.BatchNorm2d(fuera), nn.ReLU(inplace=True))


@registrar("imagen_anomalias")
class TareaAnomalias(Tarea):

    def datos(self):
        raiz = Path(self.cfg.datos.ruta)
        normales = _listar(raiz / "normal" if (raiz / "normal").is_dir() else raiz)
        if not normales:
            raise SystemExit(f"No hay imágenes normales en {raiz}")

        carpeta_raras = self.cfg.anomalias.carpeta_anomalas or (
            "anomalas" if (raiz / "anomalas").is_dir() else "")
        self.raras = _listar(raiz / carpeta_raras) if carpeta_raras else []

        aleatorio = random.Random(self.cfg.semilla)
        aleatorio.shuffle(normales)
        if self.cfg.datos.limite:
            normales = normales[:self.cfg.datos.limite]
        corte = max(1, int(len(normales) * self.cfg.datos.val_proporcion))
        train, val = normales[corte:], normales[:corte]

        comunes = dict(num_workers=self.cfg.datos.workers, collate_fn=_juntar)
        cargar = lambda datos, mezclar, raras: DataLoader(  # noqa: E731
            DatasetAnomalias(datos, raras, self.cfg, mezclar),
            batch_size=self.cfg.datos.batch, shuffle=mezclar,
            drop_last=mezclar and len(datos) > self.cfg.datos.batch, **comunes)

        print(f"anomalías: {len(train)} normales para entrenar · {len(val)} para validar"
              + (f" · {len(self.raras)} anómalas SOLO para medir" if self.raras else
                 " · sin ejemplos anómalos (no se podrá calcular el AUC)"))
        return (cargar(train, True, []), cargar(val, False, self.raras),
                InfoDatos(clases=["normal", "anomala"], conteo=[len(train), len(self.raras)],
                          n_train=len(train), n_val=len(val) + len(self.raras)))

    def modelo(self, info: InfoDatos):
        a = self.cfg.anomalias
        modelo = Autocodificador(a.canales, a.cuello, a.tam_img)
        print(f"autocodificador: cuello de {a.cuello} números · "
              f"{sum(p.numel() for p in modelo.parameters()) / 1e6:.2f} M parámetros")
        return modelo

    def criterio(self, info: InfoDatos, dispositivo):
        return nn.L1Loss().to(dispositivo)

    def paso(self, modelo, lote, criterio, dispositivo, entrenando: bool,
             espejo: bool = False) -> Paso:
        x, etiquetas = lote
        x = x.to(dispositivo, non_blocking=True)
        reconstruida = modelo(x)
        # Entrenar SOLO con lo normal: si aprendiera a reconstruir lo raro, no lo detectaría
        perdida = criterio(reconstruida, x)
        error = (reconstruida - x).abs().mean(dim=[1, 2, 3]).detach()
        return Paso(perdida=perdida, logits=error, objetivos=etiquetas.to(dispositivo),
                    datos_extra={"error": error, "etiqueta": etiquetas})

    def evaluador(self, info: InfoDatos):
        return EvaluadorAnomalias(self.cfg.anomalias.percentil)

    def descripcion(self) -> str:
        return f"autocodificador (cuello {self.cfg.anomalias.cuello})"

    def exportar_extra(self) -> dict:
        return {"arquitectura": "autocodificador", "tam_img": self.cfg.anomalias.tam_img,
                "canales": self.cfg.anomalias.canales, "cuello": self.cfg.anomalias.cuello}


class DatasetAnomalias(Dataset):
    def __init__(self, normales, raras, cfg, entrenando):
        self.rutas = [(p, 0) for p in normales] + [(p, 1) for p in raras]
        self.cfg = cfg
        self.entrenando = entrenando

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, indice):
        ruta, etiqueta = self.rutas[indice]
        with Image.open(ruta) as bruta:
            imagen = bruta.convert("RGB")
        lado = self.cfg.anomalias.tam_img
        imagen = TF.center_crop(TF.resize(imagen, lado), [lado, lado])
        if self.entrenando and random.random() < 0.5:
            imagen = TF.hflip(imagen)
        return TF.to_tensor(imagen), etiqueta


def _juntar(lote):
    xs, ys = zip(*lote)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def _listar(carpeta: Path) -> list[Path]:
    if not carpeta.is_dir():
        return []
    return sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in EXTENSIONES)


class EvaluadorAnomalias:
    """Mide el error sobre lo normal y, si hay ejemplos raros, si sabe separarlos."""

    def __init__(self, percentil: int):
        self.percentil = percentil
        self.metrica_objetivo = "acc"
        self.reiniciar()

    def reiniciar(self) -> None:
        self.errores, self.etiquetas = [], []
        self._perdida, self._n = 0.0, 0

    def actualizar(self, logits=None, objetivos=None, perdida=None, subgrupos=None,
                   datos_extra=None) -> None:
        if perdida is not None:
            self._perdida += perdida
            self._n += 1
        if not datos_extra:
            return
        self.errores += datos_extra["error"].cpu().numpy().tolist()
        self.etiquetas += datos_extra["etiqueta"].cpu().numpy().tolist()

    def resumen(self, calibrar: bool = False, curvas: bool = True) -> dict:
        if not self.errores:
            return {}
        errores = np.array(self.errores)
        etiquetas = np.array(self.etiquetas)
        normales = errores[etiquetas == 0]
        umbral = float(np.percentile(normales, self.percentil)) if len(normales) else 0.0

        salida = {"perdida": self._perdida / max(1, self._n), "n": len(errores),
                  "umbral": umbral, "error_normal": float(normales.mean()) if len(normales) else 0.0}

        if (etiquetas == 1).any():
            auc = _auc(errores, etiquetas)
            detectadas = float((errores[etiquetas == 1] > umbral).mean())
            falsas = float((normales > umbral).mean())
            salida.update({"acc": auc, "acc_balanceada": auc, "auc": auc,
                           "detectadas": detectadas, "falsas_alarmas": falsas,
                           "error_anomalo": float(errores[etiquetas == 1].mean()),
                           "texto": f"AUC {auc:.4f} · detecta el {detectadas * 100:.0f} % de "
                                    f"las anomalías con {falsas * 100:.1f} % de falsas alarmas\n"
                                    f"error medio: normal {normales.mean():.4f} · "
                                    f"anómalo {errores[etiquetas == 1].mean():.4f}"})
        else:
            # Sin ejemplos raros solo se puede seguir el error de reconstrucción
            salida.update({"acc": 1.0 / (1.0 + normales.mean()),
                           "acc_balanceada": 1.0 / (1.0 + normales.mean()),
                           "texto": f"error de reconstrucción {normales.mean():.4f} · "
                                    f"umbral (percentil {self.percentil}) {umbral:.4f}"})
        return salida

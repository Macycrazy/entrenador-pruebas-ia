"""Vigilancia térmica de la GPU durante el entrenamiento.

La tarjeta ya se protege sola bajando frecuencias, así que esto no es imprescindible:
sirve para enterarte (registro de temperatura y consumo por época) y para cortar de
forma limpia —guardando el checkpoint— si algo va mal de verdad: un ventilador parado,
un filtro tapado o una tarde demasiado calurosa.

Lee por tres vías, en orden, y si ninguna funciona se desactiva sin molestar.
"""

from __future__ import annotations

import subprocess
import time


class SobrecalentamientoGPU(RuntimeError):
    """Se lanza cuando la GPU pasa del límite varias lecturas seguidas."""


class VigilanteGPU:
    def __init__(self, temp_max: int = 85, tolerancia: int = 3, activo: bool = True,
                 cada_segundos: float = 20.0):
        self.temp_max = temp_max
        self.tolerancia = tolerancia
        self.cada = cada_segundos
        self.metodo = self._elegir_metodo() if activo else None
        self._seguidas = 0
        self._ultima = 0.0
        self.reiniciar()
        if activo and self.metodo is None:
            print("AVISO: no se puede leer la temperatura de la GPU; vigilancia desactivada.")

    # ------------------------------------------------------------------ lectura

    def _elegir_metodo(self) -> str | None:
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            torch.cuda.temperature(0)
            return "torch"
        except Exception:  # noqa: BLE001 - versiones sin soporte NVML
            pass
        try:
            self._consultar_smi()
            return "smi"
        except Exception:  # noqa: BLE001 - sin nvidia-smi en el PATH
            return None

    def _consultar_smi(self) -> tuple[float, float, float]:
        salida = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
        return interpretar_smi(salida)

    def leer(self) -> dict | None:
        """Devuelve {'temp', 'potencia', 'uso'} o None si no hay lectura disponible."""
        if self.metodo == "torch":
            import torch
            return {"temp": float(torch.cuda.temperature(0)),
                    "potencia": float(torch.cuda.power_draw(0)) / 1000.0,
                    "uso": float(torch.cuda.utilization(0))}
        if self.metodo == "smi":
            try:
                temp, potencia, uso = self._consultar_smi()
            except Exception:  # noqa: BLE001 - una lectura fallida no debe parar nada
                return None
            return {"temp": temp, "potencia": potencia, "uso": uso}
        return None

    # ------------------------------------------------------------------ control

    def comprobar(self) -> dict | None:
        """Llamar a menudo: solo mide de verdad cada `cada` segundos."""
        if self.metodo is None:
            return None
        ahora = time.monotonic()
        if ahora - self._ultima < self.cada:
            return None
        self._ultima = ahora

        lectura = self.leer()
        if lectura is None:
            return None

        self.lecturas += 1
        self.temp_suma += lectura["temp"]
        self.temp_pico = max(self.temp_pico, lectura["temp"])
        self.potencia_pico = max(self.potencia_pico, lectura["potencia"])

        if lectura["temp"] >= self.temp_max:
            self._seguidas += 1
            print(f"\n  AVISO: GPU a {lectura['temp']:.0f} °C "
                  f"(límite {self.temp_max} °C) — aviso {self._seguidas}/{self.tolerancia}")
            if self._seguidas >= self.tolerancia:
                raise SobrecalentamientoGPU(
                    f"la GPU lleva {self._seguidas} lecturas por encima de "
                    f"{self.temp_max} °C (pico {self.temp_pico:.0f} °C)")
        else:
            self._seguidas = 0
        return lectura

    def resumen(self) -> dict:
        if not self.lecturas:
            return {}
        return {"temp_media": round(self.temp_suma / self.lecturas, 1),
                "temp_pico": round(self.temp_pico, 1),
                "potencia_pico": round(self.potencia_pico, 1)}

    def reiniciar(self) -> None:
        self.lecturas = 0
        self.temp_suma = 0.0
        self.temp_pico = 0.0
        self.potencia_pico = 0.0


def interpretar_smi(salida: str) -> tuple[float, float, float]:
    """Interpreta la primera línea de nvidia-smi --format=csv,noheader,nounits.

    Los campos no disponibles vienen como '[N/A]' y se traducen a 0.
    """
    linea = next(l for l in salida.strip().splitlines() if l.strip())
    partes = [p.strip() for p in linea.split(",")]
    valores = []
    for parte in partes[:3]:
        try:
            valores.append(float(parte))
        except ValueError:
            valores.append(0.0)
    while len(valores) < 3:
        valores.append(0.0)
    return valores[0], valores[1], valores[2]

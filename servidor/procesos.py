"""Lanza entrenamientos y utilidades desde el panel, con la salida en vivo.

Cada proceso corre como un subproceso independiente del servidor: si cierras el panel
o recargas la página, el entrenamiento sigue. La salida se guarda en memoria y la vista
la va pidiendo por trozos.

Seguridad: solo se pueden lanzar los scripts de esta lista blanca, los argumentos se
pasan como lista (nunca por shell) y las rutas de configuración tienen que estar dentro
de configs/. Aun así, esto ejecuta código en tu máquina: no expongas el panel a la red
con --host 0.0.0.0 salvo que sepas lo que haces.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
MAX_LINEAS = 4000

ACCIONES = {
    "entrenar": ("entrenar.py", "Entrenamiento"),
    "estimar": ("estimar_tiempo.py", "Estimación de tiempo"),
    "evaluar": ("evaluar.py", "Evaluación"),
    "exportar": ("exportar.py", "Exportación"),
    "buscar": ("buscar.py", "Búsqueda de hiperparámetros"),
    "descargar_dataset": ("preparacion/descargar_dataset.py", "Descarga de dataset"),
    "descargar_rostros": ("preparacion/descargar_rostros.py", "Descarga de rostros"),
    "preparar_datos": ("preparacion/preparar_datos.py", "Preparación de datos"),
    "comprobar_gpu": ("preparacion/comprobar_gpu.py", "Comprobación de GPU"),
    "sintetico_deteccion": ("preparacion/generar_deteccion_sintetica.py", "Datos sintéticos"),
    "sintetico_segmentacion": ("preparacion/generar_segmentacion_sintetica.py", "Datos sintéticos"),
    "sintetico_audio": ("preparacion/generar_audio_sintetico.py", "Datos sintéticos"),
    "sintetico_texto": ("preparacion/generar_texto_sintetico.py", "Datos sintéticos"),
    "sintetico_llm": ("preparacion/generar_instrucciones_sintetico.py", "Datos sintéticos"),
}

# clave.anidada=valor, sin espacios ni metacaracteres
PATRON_OVERRIDE = re.compile(r"^[a-zA-Z_][\w]*(\.[a-zA-Z_][\w]*)*=[^\s;|&$`<>]+$")


@dataclass
class Proceso:
    id: str
    accion: str
    descripcion: str
    comando: list[str]
    inicio: float
    popen: subprocess.Popen | None = None
    lineas: list[str] = field(default_factory=list)
    estado: str = "ejecutando"       # ejecutando | terminado | fallido | detenido
    codigo: int | None = None

    def resumen(self) -> dict:
        return {"id": self.id, "accion": self.accion, "descripcion": self.descripcion,
                "comando": " ".join(self.comando[1:]), "estado": self.estado,
                "codigo": self.codigo, "segundos": round(time.time() - self.inicio, 1),
                "lineas": len(self.lineas)}


_PROCESOS: dict[str, Proceso] = {}
_CANDADO = threading.Lock()


def lanzar(accion: str, argumentos: list[str], descripcion: str = "") -> Proceso:
    if accion not in ACCIONES:
        raise ValueError(f"Acción '{accion}' no permitida")
    script, titulo = ACCIONES[accion]
    comando = [sys.executable, str(RAIZ / script), *_validar(argumentos)]

    proceso = Proceso(id=uuid.uuid4().hex[:8], accion=accion,
                      descripcion=descripcion or titulo, comando=comando,
                      inicio=time.time())
    proceso.popen = subprocess.Popen(
        comando, cwd=str(RAIZ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=_entorno())

    with _CANDADO:
        _PROCESOS[proceso.id] = proceso
    threading.Thread(target=_leer, args=(proceso,), daemon=True).start()
    return proceso


def _entorno() -> dict:
    import os
    entorno = dict(os.environ)
    entorno["PYTHONUNBUFFERED"] = "1"      # sin esto la salida llega a trozos gigantes
    entorno["PYTHONIOENCODING"] = "utf-8"
    return entorno


def _validar(argumentos: list[str]) -> list[str]:
    """Deja pasar solo banderas conocidas, rutas dentro del proyecto y overrides sanos."""
    limpios = []
    esperando_valor = None

    for bruto in argumentos:
        argumento = str(bruto).strip()
        if not argumento:
            continue

        if esperando_valor == "config":
            ruta = (RAIZ / argumento).resolve()
            if not str(ruta).startswith(str(RAIZ / "configs")) or not ruta.exists():
                raise ValueError(f"Configuración no permitida: {argumento}")
            limpios.append(str(ruta.relative_to(RAIZ)))
            esperando_valor = None
            continue
        if esperando_valor == "set":
            if not PATRON_OVERRIDE.match(argumento):
                raise ValueError(f"Override inválido: {argumento}")
            limpios.append(argumento)
            esperando_valor = None
            continue
        if esperando_valor == "libre":
            if re.search(r"[;|&$`<>]", argumento):
                raise ValueError(f"Argumento con caracteres no permitidos: {argumento}")
            limpios.append(argumento)
            esperando_valor = None
            continue

        if argumento == "--config":
            esperando_valor = "config"
        elif argumento == "--set":
            esperando_valor = "set"
        elif argumento.startswith("--"):
            if not re.fullmatch(r"--[a-z0-9-]+", argumento):
                raise ValueError(f"Bandera inválida: {argumento}")
            esperando_valor = "libre"
        else:
            if re.search(r"[;|&$`<>]", argumento):
                raise ValueError(f"Argumento con caracteres no permitidos: {argumento}")
            limpios.append(argumento)
            continue
        limpios.append(argumento)

    return limpios


def _leer(proceso: Proceso) -> None:
    try:
        for linea in proceso.popen.stdout:
            # El progreso usa \r para reescribir la línea: nos quedamos con el último trozo
            texto = linea.rstrip("\n").split("\r")[-1]
            if texto.strip():
                proceso.lineas.append(texto)
                if len(proceso.lineas) > MAX_LINEAS:
                    del proceso.lineas[:len(proceso.lineas) - MAX_LINEAS]
    finally:
        proceso.codigo = proceso.popen.wait()
        if proceso.estado != "detenido":
            proceso.estado = "terminado" if proceso.codigo == 0 else "fallido"


def obtener(identificador: str) -> Proceso | None:
    return _PROCESOS.get(identificador)


def listar() -> list[dict]:
    with _CANDADO:
        return [p.resumen() for p in sorted(_PROCESOS.values(),
                                            key=lambda p: -p.inicio)]


def salida(identificador: str, desde: int = 0) -> dict:
    proceso = obtener(identificador)
    if proceso is None:
        raise KeyError(identificador)
    lineas = proceso.lineas[desde:]
    return {**proceso.resumen(), "desde": desde,
            "hasta": desde + len(lineas), "nuevas": lineas}


def parar(identificador: str) -> dict:
    proceso = obtener(identificador)
    if proceso is None:
        raise KeyError(identificador)
    if proceso.estado == "ejecutando" and proceso.popen:
        proceso.estado = "detenido"
        proceso.popen.terminate()
        try:
            proceso.popen.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proceso.popen.kill()
        proceso.lineas.append("--- detenido desde el panel ---")
    return proceso.resumen()


def limpiar() -> int:
    """Quita del historial los procesos que ya no corren."""
    with _CANDADO:
        idos = [i for i, p in _PROCESOS.items() if p.estado != "ejecutando"]
        for identificador in idos:
            del _PROCESOS[identificador]
    return len(idos)


def hay_entrenando() -> bool:
    return any(p.estado == "ejecutando" and p.accion == "entrenar"
               for p in _PROCESOS.values())

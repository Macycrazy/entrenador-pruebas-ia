#!/usr/bin/env python3
"""Panel de IA: una vista por tarea, cada una con su propia página y su propia API.

    python servidor/app.py                 # http://127.0.0.1:8000
    python servidor/app.py --puerto 8010

El servidor solo lee modelos ya entrenados; el entrenamiento va aparte
(`python entrenar.py --config configs/<tarea>.yaml`). Las tareas sin modelo siguen
teniendo su página: explica cómo entrenarlas.
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

import torch
from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from servidor import ejemplos as banco_ejemplos, procesos, servicios
from servidor.vistas import FAMILIAS, PAGINAS, POR_SLUG, VISTAS

app = FastAPI(title="Panel de IA", docs_url="/api/docs")
app.mount("/estaticos", StaticFiles(directory=AQUI / "estaticos"), name="estaticos")
plantillas = Jinja2Templates(directory=str(AQUI / "plantillas"))
# las familias las usan la cabecera y la portada: global, para no pasarlas por ruta
plantillas.env.globals["familias"] = FAMILIAS


def _servicio(slug: str):
    vista = POR_SLUG.get(slug)
    if vista is None:
        raise HTTPException(404, f"No existe la vista '{slug}'")
    try:
        return servicios.obtener(vista)
    except FileNotFoundError:
        raise HTTPException(503, f"No hay modelo entrenado para '{vista.titulo}'. "
                                 f"Entrénalo con: python entrenar.py --config {vista.config}")
    except SystemExit as error:  # los módulos de tarea lo usan para "falta tal librería"
        raise HTTPException(503, str(error) or "Falta una dependencia") from None
    except Exception as error:  # noqa: BLE001 - dependencia ausente, checkpoint roto…
        raise HTTPException(503, f"{type(error).__name__}: {error}") from error


async def _leer(archivo: UploadFile) -> bytes:
    datos = await archivo.read()
    if not datos:
        raise HTTPException(400, "Archivo vacío")
    return datos


# ------------------------------------------------------------------ páginas

@app.get("/", response_class=HTMLResponse)
def inicio(request: Request):
    # Starlette moderno exige el orden (request, plantilla, contexto).
    return plantillas.TemplateResponse(request, "inicio.html", {
        "vistas": VISTAS, "paginas": PAGINAS, "activa": "inicio",
        # se calcula aquí: en la plantilla, `selectattr("disponible")` recibiría el
        # método sin llamar (siempre verdadero) y contaría todas las tareas
        "listas": sum(1 for v in VISTAS if v.disponible()),
        # entrenados aquí = con checkpoint propio; el resto de las listas van con un
        # modelo base descargado, que funciona igual pero no lo hemos entrenado nosotros
        "entrenados": sum(1 for v in VISTAS if v.entrenada()),
        "por_entrenar": sum(1 for v in VISTAS if v.nivel() == "sin"),
        "dispositivo": "GPU" if torch.cuda.is_available() else "CPU",
    })


@app.get("/{slug}", response_class=HTMLResponse)
def pagina(request: Request, slug: str):
    vista = POR_SLUG.get(slug)
    if vista is None:
        raise HTTPException(404, "Página no encontrada")
    return plantillas.TemplateResponse(request, vista.archivo_plantilla, {
        "vistas": VISTAS, "paginas": PAGINAS, "activa": slug, "vista": vista,
        "configs": _configs(),
        "experimentos": _experimentos(),
        # la guía enseña las mismas cifras que la portada, leídas en vivo
        "listas": sum(1 for v in VISTAS if v.disponible()),
        "entrenados": sum(1 for v in VISTAS if v.entrenada()),
        "por_entrenar": sum(1 for v in VISTAS if v.nivel() == "sin"),
    })


# ------------------------------------------------------------------ estado

@app.get("/api/memoria")
def memoria():
    """Qué modelos están cargados ahora mismo y cuánta memoria queda."""
    datos = {"cargados": servicios.cargados(), "maximo": servicios.MAX_CARGADOS}
    if torch.cuda.is_available():
        libre, total = torch.cuda.mem_get_info()
        datos["vram_libre_gb"] = round(libre / 1024**3, 2)
        datos["vram_total_gb"] = round(total / 1024**3, 2)
    return JSONResponse(datos)


@app.post("/api/memoria/liberar")
def liberar_memoria():
    return JSONResponse({"descargados": servicios.vaciar()})


@app.get("/api/vistas")
def listar_vistas():
    from servidor.vistas import estado
    return JSONResponse(estado())


@app.get("/api/{slug}/ejemplos")
def ejemplos_vista(slug: str):
    """Material de muestra para probar la vista sin traer nada de casa."""
    if slug not in POR_SLUG:
        raise HTTPException(404, "Vista desconocida")
    return JSONResponse(banco_ejemplos.listar(slug))


@app.get("/api/{slug}/ejemplo/{indice}")
def ejemplo_vista(slug: str, indice: int):
    ruta = banco_ejemplos.archivo(slug, indice)
    if ruta is None or not ruta.exists():
        raise HTTPException(404, "Ese ejemplo no existe")
    from fastapi.responses import FileResponse
    return FileResponse(ruta)


@app.get("/api/{slug}/estado")
def estado_vista(slug: str):
    vista = POR_SLUG.get(slug)
    if vista is None:
        raise HTTPException(404, "Vista desconocida")
    try:
        return JSONResponse(servicios.obtener(vista).info())
    except HTTPException:
        raise
    except FileNotFoundError:
        return JSONResponse({"listo": False, "config": vista.config,
                             "detalle": "sin modelo entrenado"}, status_code=503)
    except SystemExit as error:  # mismo caso que en _servicio()
        return JSONResponse({"listo": False, "config": vista.config,
                             "detalle": str(error) or "Falta una dependencia"},
                            status_code=503)
    except Exception as error:  # noqa: BLE001
        return JSONResponse({"listo": False, "config": vista.config,
                             "detalle": f"{type(error).__name__}: {error}"}, status_code=503)


# ------------------------------------------------------------------ rostros

@app.get("/api/rostros/galeria")
def galeria():
    servicio = _servicio("rostros")
    if servicio.galeria is None:
        return JSONResponse({"personas": [], "umbral": servicio.cfg.rostros.umbral_similitud})
    return JSONResponse({"personas": servicio.galeria["nombres"],
                         "umbral": servicio.galeria["umbral"]})


@app.post("/api/rostros/identificar")
async def identificar(imagen: UploadFile = File(...), umbral: float | None = Query(None)):
    return JSONResponse(_servicio("rostros").identificar(await _leer(imagen), umbral))


@app.post("/api/rostros/verificar")
async def verificar(imagen: UploadFile = File(...), contra: UploadFile = File(...),
                    umbral: float | None = Query(None)):
    servicio = _servicio("rostros")
    return JSONResponse(servicio.verificar(await _leer(imagen), await _leer(contra), umbral))


@app.post("/api/rostros/inscribir")
async def inscribir(nombre: str = Query(..., min_length=1, max_length=80),
                    imagenes: list[UploadFile] = File(...)):
    servicio = _servicio("rostros")
    datos = [await _leer(i) for i in imagenes]
    try:
        return JSONResponse(servicio.inscribir(nombre.strip(), datos))
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.delete("/api/rostros/inscritos/{nombre}")
def olvidar(nombre: str):
    try:
        return JSONResponse(_servicio("rostros").olvidar(nombre))
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


# ------------------------------------------------------------------ audio y texto

@app.post("/api/audio/predecir")
async def predecir_audio(audio: UploadFile = File(...)):
    datos = await _leer(audio)
    return JSONResponse(_servicio("audio").predecir(datos, audio.filename or "audio.wav"))


@app.post("/api/texto/predecir")
def predecir_texto(cuerpo: dict = Body(...)):
    texto = (cuerpo.get("texto") or "").strip()
    if not texto:
        raise HTTPException(400, "Texto vacío")
    return JSONResponse(_servicio("texto").predecir(texto))


@app.post("/api/llm/responder")
def responder_llm(cuerpo: dict = Body(...)):
    instruccion = (cuerpo.get("instruccion") or "").strip()
    if not instruccion:
        raise HTTPException(400, "Falta la instrucción")
    return JSONResponse(_servicio("llm").responder(
        instruccion, cuerpo.get("entrada", ""),
        int(cuerpo.get("tokens", 200)), float(cuerpo.get("temperatura", 0.7))))


@app.post("/api/transcripcion/predecir")
async def transcribir(audio: UploadFile = File(...), idioma: str | None = Query(None)):
    datos = await _leer(audio)
    return JSONResponse(_servicio("transcripcion").transcribir(
        datos, audio.filename or "audio.wav", idioma))


@app.post("/api/imagenes/indexar")
def indexar_imagenes(cuerpo: dict = Body(...)):
    servicio = _servicio("imagenes")
    try:
        return JSONResponse(servicio.indexar(cuerpo.get("carpeta", ""),
                                             int(cuerpo.get("limite", 500))))
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/imagenes/buscar")
def buscar_imagenes(cuerpo: dict = Body(...)):
    texto = (cuerpo.get("texto") or "").strip()
    if not texto:
        raise HTTPException(400, "Falta la descripción")
    return JSONResponse(_servicio("imagenes").buscar(texto, cuerpo.get("cuantos")))


@app.post("/api/tabular/predecir")
def tabular(cuerpo: dict = Body(...)):
    return JSONResponse(_servicio("tabular").predecir(cuerpo.get("fila") or {}))


@app.post("/api/series/predecir")
def series(cuerpo: dict = Body(...)):
    valores = cuerpo.get("valores") or []
    try:
        return JSONResponse(_servicio("series").predecir([float(v) for v in valores]))
    except (ValueError, TypeError) as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/ner/predecir")
def ner(cuerpo: dict = Body(...)):
    texto = (cuerpo.get("texto") or "").strip()
    if not texto:
        raise HTTPException(400, "Texto vacío")
    return JSONResponse(_servicio("ner").predecir(texto))


@app.get("/api/busqueda/estado_indice")
def estado_indice():
    servicio = _servicio("busqueda")
    return JSONResponse({"fragmentos": len(servicio.documentos),
                         "origenes": sorted({d["origen"] for d in servicio.documentos})})


@app.post("/api/busqueda/indexar")
async def indexar(archivo: UploadFile | None = File(None), cuerpo: dict | None = Body(None)):
    servicio = _servicio("busqueda")
    if archivo is not None:
        datos = await _leer(archivo)
        texto = datos.decode("utf-8", errors="ignore")
        origen = archivo.filename or "documento.txt"
    else:
        texto = (cuerpo or {}).get("texto", "")
        origen = (cuerpo or {}).get("origen", "pegado")
    try:
        return JSONResponse(servicio.indexar(texto, origen))
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/busqueda/buscar")
def buscar(cuerpo: dict = Body(...)):
    pregunta = (cuerpo.get("pregunta") or "").strip()
    if not pregunta:
        raise HTTPException(400, "Falta la pregunta")
    return JSONResponse(_servicio("busqueda").buscar(pregunta, cuerpo.get("cuantos")))


@app.delete("/api/busqueda/documentos/{origen}")
def olvidar_documento(origen: str):
    return JSONResponse(_servicio("busqueda").olvidar_documento(origen))


@app.post("/api/ocr/predecir")
async def ocr(imagen: UploadFile = File(...), umbral: float = Query(0.35, ge=0.0, le=1.0)):
    return JSONResponse(_servicio("ocr").predecir(await _leer(imagen), umbral))


@app.post("/api/seguimiento/predecir")
async def seguimiento(imagen: UploadFile = File(...), umbral: float = Query(0.6, ge=0.1, le=1.0),
                      solo: str | None = Query(None)):
    return JSONResponse(_servicio("seguimiento").predecir(await _leer(imagen), umbral, solo))


@app.post("/api/anomalias/predecir")
async def anomalias(imagen: UploadFile = File(...), umbral: float | None = Query(None)):
    resultado = _servicio("anomalias").predecir(await _leer(imagen), umbral)
    for clave in ("mapa", "reconstruida", "entrada"):
        resultado[clave] = "data:image/png;base64," + base64.b64encode(resultado[clave]).decode()
    return JSONResponse(resultado)


@app.post("/api/pose/predecir")
async def pose(imagen: UploadFile = File(...), umbral: float = Query(0.8, ge=0.1, le=1.0)):
    return JSONResponse(_servicio("pose").predecir(await _leer(imagen), umbral))


@app.post("/api/profundidad/predecir")
async def profundidad(imagen: UploadFile = File(...)):
    resultado = _servicio("profundidad").predecir(await _leer(imagen))
    for clave in ("png", "gris"):
        resultado[clave] = "data:image/png;base64," + base64.b64encode(resultado[clave]).decode()
    resultado["mapa"] = resultado.pop("png")
    return JSONResponse(resultado)


@app.post("/api/generacion/generar")
def generar(cuerpo: dict = Body(...)):
    from fastapi.responses import Response

    prompt = (cuerpo.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "Falta la descripción de lo que hay que dibujar")
    resultado = _servicio("generacion").generar(
        prompt, cuerpo.get("pasos"), cuerpo.get("guia"), cuerpo.get("negativo"),
        cuerpo.get("semilla"), cuerpo.get("lado"))
    return Response(content=resultado["png"], media_type="image/png", headers={
        "X-Ms": str(round(resultado["ms"])), "X-Pasos": str(resultado["pasos"]),
    })


# ------------------------------------------------------------------ voz

@app.post("/api/voz/hablar")
def hablar(cuerpo: dict = Body(...)):
    from fastapi.responses import Response

    texto = (cuerpo.get("texto") or "").strip()
    if not texto:
        raise HTTPException(400, "Texto vacío")
    resultado = _servicio("voz").hablar(texto, cuerpo.get("voz"))
    return Response(content=resultado["wav"], media_type="audio/wav", headers={
        "X-Segundos": str(resultado["segundos"]), "X-Ms": str(round(resultado["ms"])),
    })


@app.post("/api/voz/clonar")
async def clonar_voz(nombre: str = Query(..., min_length=1, max_length=60),
                     audio: UploadFile = File(...)):
    datos = await _leer(audio)
    servicio = _servicio("voz")
    try:
        return JSONResponse(servicio.clonar(
            nombre.strip(), datos, Path(audio.filename or "ref.wav").suffix or ".wav"))
    except SystemExit as error:
        raise HTTPException(503, str(error)) from error


@app.delete("/api/voz/voces/{nombre}")
def olvidar_voz(nombre: str):
    try:
        return JSONResponse(_servicio("voz").olvidar_voz(nombre))
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


# ------------------------------------------------------------------ procesos

@app.get("/api/procesos")
def listar_procesos():
    return JSONResponse({"procesos": procesos.listar(),
                         "entrenando": procesos.hay_entrenando()})


@app.post("/api/procesos/lanzar")
def lanzar_proceso(cuerpo: dict = Body(...)):
    accion = cuerpo.get("accion", "")
    argumentos = cuerpo.get("argumentos", [])
    if not isinstance(argumentos, list):
        raise HTTPException(400, "argumentos debe ser una lista")
    try:
        proceso = procesos.lanzar(accion, argumentos, cuerpo.get("descripcion", ""))
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except OSError as error:
        raise HTTPException(500, f"No se pudo lanzar: {error}") from error
    return JSONResponse(proceso.resumen())


@app.get("/api/procesos/{identificador}/salida")
def salida_proceso(identificador: str, desde: int = Query(0, ge=0)):
    try:
        return JSONResponse(procesos.salida(identificador, desde))
    except KeyError:
        raise HTTPException(404, "Proceso desconocido") from None


@app.post("/api/procesos/{identificador}/parar")
def parar_proceso(identificador: str):
    try:
        return JSONResponse(procesos.parar(identificador))
    except KeyError:
        raise HTTPException(404, "Proceso desconocido") from None


@app.post("/api/procesos/limpiar")
def limpiar_procesos():
    return JSONResponse({"quitados": procesos.limpiar()})


@app.get("/api/experimentos")
def listar_experimentos():
    return JSONResponse({"experimentos": _experimentos()})


@app.get("/api/experimentos/{nombre}/historial")
def historial(nombre: str):
    """Las columnas del historial.csv, listas para dibujar la curva de entrenamiento."""
    import csv as csv_mod

    ruta = _dentro(RAIZ / "experimentos" / nombre / "historial.csv", RAIZ / "experimentos")
    if not ruta.exists():
        return JSONResponse({"columnas": [], "filas": []})
    with ruta.open() as f:
        filas = list(csv_mod.DictReader(f))
    if not filas:
        return JSONResponse({"columnas": [], "filas": []})

    columnas = [c for c in filas[0] if c != "epoca"]
    series = {c: [_numero(fila.get(c)) for fila in filas] for c in columnas}
    return JSONResponse({"columnas": columnas, "epocas": len(filas), "series": series})


@app.get("/api/experimentos/{nombre}/descargar")
def descargar_modelo(nombre: str, formato: str = Query("pt")):
    from fastapi.responses import FileResponse

    archivos = {"pt": "mejor.pt", "onnx": "modelo.onnx", "int8": "modelo_int8.pt",
                "historial": "historial.csv", "metricas": "metricas.json",
                "config": "config.yaml"}
    if formato not in archivos:
        raise HTTPException(400, f"Formato desconocido. Opciones: {', '.join(archivos)}")
    ruta = _dentro(RAIZ / "experimentos" / nombre / archivos[formato], RAIZ / "experimentos")
    if not ruta.exists():
        raise HTTPException(404, f"No existe {archivos[formato]} en {nombre}. "
                                 "Para ONNX o INT8, expórtalo primero.")
    return FileResponse(ruta, filename=f"{nombre}_{ruta.name}",
                        media_type="application/octet-stream")


# ------------------------------------------------------------------ datos

@app.get("/api/datos/resumen")
def resumen_datos():
    """Qué datasets hay en disco, con sus clases y su reparto por subgrupo."""
    import collections
    import csv as csv_mod

    salida = []
    for carpeta in sorted(RAIZ.glob("datos*")):
        if not carpeta.is_dir() or carpeta.name.startswith("."):
            continue
        raiz = carpeta / "train" if (carpeta / "train").is_dir() else carpeta
        clases = {}
        for sub in sorted(p for p in raiz.iterdir() if p.is_dir()):
            if sub.name in ("imagenes", "images", "etiquetas", "labels", "mascaras", "masks"):
                # Detección y segmentación no van por carpetas de clase: se cuentan las
                # imágenes, que si no el panel enseñaba "0 archivos" y parecía roto.
                imagenes = raiz / ("imagenes" if (raiz / "imagenes").is_dir() else "images")
                n = sum(1 for _ in imagenes.iterdir()) if imagenes.is_dir() else 0
                clases = {"imágenes anotadas (sin carpetas de clase)": n}
                break
            clases[sub.name] = sum(1 for _ in sub.iterdir())
        if not clases:
            continue

        subgrupos = {}
        meta = carpeta / "metadatos.csv"
        if meta.exists():
            with meta.open() as f:
                filas = list(csv_mod.DictReader(f))
            for columna in [c for c in (filas[0] if filas else {}) if c not in ("archivo", "clase")]:
                subgrupos[columna] = dict(collections.Counter(
                    fila[columna] for fila in filas).most_common(12))
        salida.append({"nombre": carpeta.name, "raiz": str(raiz.relative_to(RAIZ)),
                       "clases": clases, "total": sum(clases.values()),
                       "subgrupos": subgrupos})
    return JSONResponse({"datasets": salida})


@app.get("/api/datos/muestras")
def muestras(carpeta: str = Query(...), n: int = Query(24, ge=1, le=120),
             desde: int = Query(0, ge=0)):
    import random

    directorio = _dentro(RAIZ / carpeta, RAIZ)
    if not directorio.is_dir():
        raise HTTPException(404, f"No existe {carpeta}")
    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    archivos = sorted(p for p in directorio.iterdir() if p.suffix.lower() in extensiones)
    random.Random(desde).shuffle(archivos)
    elegidas = archivos[:n]
    return JSONResponse({"total": len(archivos),
                         "rutas": [str(p.relative_to(RAIZ)) for p in elegidas]})


@app.get("/api/datos/imagen")
def imagen_dataset(ruta: str = Query(...)):
    from fastapi.responses import FileResponse

    archivo = _dentro(RAIZ / ruta, RAIZ)
    if not archivo.is_file():
        raise HTTPException(404, "No existe la imagen")
    return FileResponse(archivo)


# ------------------------------------------------------------------ comparar

@app.post("/api/comparar")
async def comparar(imagen: UploadFile = File(...), experimentos: str = Query(...)):
    """Pasa la misma imagen por varios modelos para ver la diferencia entre ellos."""
    datos = await _leer(imagen)
    resultados = []
    for nombre in [n.strip() for n in experimentos.split(",") if n.strip()][:4]:
        try:
            servicio = servicios.comparador(nombre)
            prediccion = servicio.predecir(datos)
            resultados.append({"experimento": nombre, "ok": True, **prediccion,
                               "modelo": servicio.meta.get("arquitectura"),
                               "acc_val": (servicio.meta.get("metricas") or {}).get("acc")})
        except Exception as error:  # noqa: BLE001 - un modelo roto no debe tumbar la comparación
            resultados.append({"experimento": nombre, "ok": False,
                               "detalle": f"{type(error).__name__}: {error}"})
    return JSONResponse({"resultados": resultados})


def _dentro(ruta: Path, raiz: Path) -> Path:
    """Evita que un parámetro de la URL se salga de la carpeta permitida."""
    resuelta = ruta.resolve()
    if not str(resuelta).startswith(str(raiz.resolve())):
        raise HTTPException(400, "Ruta fuera del proyecto")
    return resuelta


def _numero(valor):
    try:
        numero = float(valor)
        return None if numero != numero else numero      # NaN -> null
    except (TypeError, ValueError):
        return None


def _configs() -> list[dict]:
    """Configuraciones disponibles, marcando cuáles tienen ya sus datos en disco.

    Sin esto es facilísimo lanzar un entrenamiento que falla al segundo por no tener
    el dataset descargado.
    """
    import yaml

    salida = []
    for ruta in sorted((RAIZ / "configs").glob("*.yaml")):
        try:
            datos = (yaml.safe_load(ruta.read_text()) or {}).get("datos", {})
        except Exception:  # noqa: BLE001 - un YAML roto no debe tumbar la página
            datos = {}
        carpeta = datos.get("ruta", "datos")
        existe = (RAIZ / carpeta).exists() and any((RAIZ / carpeta).iterdir()) \
            if (RAIZ / carpeta).is_dir() else False
        salida.append({"archivo": ruta.name, "datos": str(carpeta), "listo": existe})
    salida.sort(key=lambda c: (not c["listo"], c["archivo"]))
    return salida


def _experimentos() -> list[dict]:
    """Resumen de cada carpeta de experimentos/ para la vista de entrenamiento."""
    import json

    carpeta = RAIZ / "experimentos"
    salida = []
    for directorio in sorted(carpeta.iterdir() if carpeta.exists() else []):
        if not directorio.is_dir():
            continue
        informe = directorio / "metricas.json"
        metricas = {}
        if informe.exists():
            try:
                mejor = json.loads(informe.read_text()).get("mejor", {})
                metricas = {k: round(v, 4) for k, v in mejor.items()
                            if isinstance(v, (int, float))}
            except (json.JSONDecodeError, OSError):
                pass
        # la tarea permite que el comparador solo ofrezca modelos comparables entre sí
        tarea = ""
        cfg = directorio / "config.yaml"
        if cfg.exists():
            for linea in cfg.read_text().splitlines():
                if linea.startswith("tarea:"):
                    tarea = linea.split(":", 1)[1].strip()
                    break
        salida.append({
            "nombre": directorio.name,
            "tarea": tarea,
            "tiene_modelo": (directorio / "mejor.pt").exists(),
            "metricas": metricas,
            "epocas": sum(1 for _ in (directorio / "historial.csv").open()) - 1
                      if (directorio / "historial.csv").exists() else 0,
        })
    return salida


# ------------------------------------------------------------------ imagen
#
# IMPORTANTE: este bloque va el último. FastAPI resuelve las rutas en el orden en
# que se declaran, y "/api/{slug}/predecir" capturaría también "/api/audio/predecir"
# o "/api/transcripcion/predecir", exigiéndoles un campo "imagen" que no envían.

@app.post("/api/{slug}/predecir")
async def predecir_imagen(slug: str, imagen: UploadFile = File(...),
                          detectar: bool = Query(True), max_rostros: int = Query(5, ge=1, le=20),
                          umbral: float = Query(0.5, ge=0.0, le=1.0)):
    """Clasificación de rostros, detección de objetos o segmentación, según la vista."""
    if slug not in ("genero", "atributos", "antispoofing", "deteccion", "segmentacion",
                    "superresolucion"):
        raise HTTPException(404, "Esta vista no acepta imágenes")
    servicio = _servicio(slug)
    datos = await _leer(imagen)

    if slug == "deteccion":
        return JSONResponse(servicio.predecir(datos, umbral))
    if slug == "superresolucion":
        resultado = servicio.predecir(datos)
        for clave in ("mejorada", "bicubica", "original"):
            resultado[clave] = "data:image/png;base64," + \
                base64.b64encode(resultado[clave]).decode()
        return JSONResponse(resultado)
    if slug == "segmentacion":
        resultado = servicio.predecir(datos)
        png = resultado.pop("png")
        resultado["mascara"] = "data:image/png;base64," + base64.b64encode(png).decode()
        return JSONResponse(resultado)
    return JSONResponse(servicio.predecir(datos, detectar=detectar, max_rostros=max_rostros))



# ------------------------------------------------------------------ arranque

def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="Panel de IA")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 para exponerlo en la red")
    p.add_argument("--puerto", type=int, default=8000)
    p.add_argument("--recargar", action="store_true", help="Recarga automática (desarrollo)")
    args = p.parse_args()

    listas = sum(1 for v in VISTAS if v.disponible())
    propios = sum(1 for v in VISTAS if v.entrenada())
    print(f"\n  Panel:   http://127.0.0.1:{args.puerto}")
    print(f"  Vistas:  {len(VISTAS)} demostraciones · {listas} listas para usar "
          f"({propios} entrenadas aquí)")
    for vista in VISTAS:
        marca = {"entrenado": "✓", "base": "·"}.get(vista.nivel(), " ")
        print(f"    {marca} /{vista.slug:<14} {vista.titulo}")
    if args.host not in ("127.0.0.1", "localhost"):
        print("\n  Nota: la cámara y el micrófono solo funcionan en localhost o por HTTPS.")
    print()

    uvicorn.run("servidor.app:app" if args.recargar else app,
                host=args.host, port=args.puerto, reload=args.recargar)


if __name__ == "__main__":
    main()

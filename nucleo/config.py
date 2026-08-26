"""Configuración: valores por defecto + YAML + presets + overrides de línea de comandos.

Toda opción del sistema vive en DEFECTOS. Un YAML solo escribe lo que cambia,
y `--set clave.anidada=valor` gana sobre todo lo demás.

    cfg = cargar(preset="calidad", ruta="configs/genero.yaml", overrides=["entrenamiento.epocas=30"])
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_CONFIGS = RAIZ / "configs"


class Config(dict):
    """dict con acceso por atributo: cfg.optimizador.nombre"""

    def __getattr__(self, clave):
        try:
            valor = self[clave]
        except KeyError as error:
            raise AttributeError(f"No existe la opción '{clave}'") from error
        return Config(valor) if isinstance(valor, dict) else valor

    def __setattr__(self, clave, valor):
        self[clave] = valor


# ---------------------------------------------------------------------------
# Catálogo completo de opciones. Los valores aquí son los que se usan si nadie
# los cambia; los comentarios documentan las alternativas.
# ---------------------------------------------------------------------------
DEFECTOS = {
    "tarea": "imagen_clasificacion",
    "nombre": "experimento",
    "semilla": 42,
    "determinista": False,          # True = reproducible exacto, algo más lento

    "datos": {
        "ruta": str(RAIZ / "datos"),
        "clases": None,             # None = deducir de las carpetas
        "tam_img": 224,
        "batch": 96,
        "workers": 8,
        "balanceo": "sampler",      # sampler | pesos_perdida | ninguno
        "kfold": 0,                 # 0 = usar datos/train y datos/val tal cual
        "fold": 0,
        "val_proporcion": 0.15,     # solo si kfold=0 y no hay carpeta val
        "agrupar_por": None,        # columna del CSV para no partir el mismo grupo/identidad
        "metadatos": None,          # CSV con columnas extra (edad, etnia…) para subgrupos
        "subgrupos": [],            # columnas por las que desglosar métricas
        "objetivos_extra": [],      # columnas para entrenamiento multitarea
        "deduplicar": False,        # quitar casi-duplicados por hash perceptual
        "cache_ram": False,         # cargar el dataset entero en memoria
        "limite": 0,                # 0 = sin límite (útil para pruebas rápidas)
    },

    "modelo": {
        "arquitectura": "efficientnet_v2_s",   # torchvision, "timm:nombre" o "rostro:nombre"
        "preentrenado": True,
        "dropout": 0.2,
        "drop_path": 0.0,           # stochastic depth (solo timm)
        "congelar_backbone": False,
        "cabeza": "lineal",         # lineal | arcface | cosface
        "checkpoint_gradiente": False,   # menos VRAM, ~30 % más lento
    },

    "optimizador": {
        "nombre": "adamw",          # sgd|adam|adamw|adamw_fused|nadam|radam|adamax|
                                    # rmsprop|lion|adam8bit
        "lr": 3e-4,
        "wd": 0.02,
        "momentum": 0.9,            # sgd/rmsprop
        "nesterov": True,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "lr_backbone_factor": 1.0,  # <1 = learning rate discriminativo (0.1 es habitual)
        "llrd": 0.0,                # >0 = layer-wise lr decay (0.75 típico en ViT)
        "sam": False,               # Sharpness-Aware Minimization: +generalización, 2x lento
        "sam_rho": 0.05,
        "lookahead": False,
    },

    "scheduler": {
        "nombre": "coseno",         # coseno|coseno_reinicios|onecycle|step|multistep|
                                    # exponencial|plateau|polinomial|constante
        "warmup_epocas": 0.5,
        "min_lr": 0.0,
        "step_tam": 10,             # step
        "hitos": [10, 15],          # multistep
        "gamma": 0.1,               # step/multistep/exponencial
        "paciencia_plateau": 2,
        "T0": 5,                    # coseno con reinicios
        "potencia": 1.0,            # polinomial
    },

    "perdida": {
        "nombre": "ce",             # ce|focal|bce|arcface
        "suavizado": 0.05,          # label smoothing
        "focal_gamma": 2.0,
        "pesos_clase": False,       # pondera la loss por frecuencia inversa
        "arcface_margen": 0.5,
        "arcface_escala": 30.0,
        "destilacion": {
            "activa": False,
            "profesor": None,       # ruta a un checkpoint entrenado
            "temperatura": 4.0,
            "alfa": 0.5,
        },
    },

    "aumentos": {
        "politica": "basica",       # ninguna|basica|randaugment|trivialaugment|autoaugment|augmix
        "randaugment_n": 2,
        "randaugment_m": 9,
        "flip": 0.5,
        "recorte_escala": [0.65, 1.0],
        "rotacion": 12,
        "color": [0.3, 0.3, 0.25, 0.03],   # brillo, contraste, saturación, tono
        "grises": 0.05,
        "borrado": 0.25,            # random erasing
        "webcam": 0.0,              # 0-1: desenfoque, ruido, JPEG y bajada de resolución
        "mixup": 0.0,               # alfa de mixup (0.2 típico)
        "cutmix": 0.0,              # alfa de cutmix (1.0 típico)
        "mixcut_prob": 0.5,         # probabilidad de aplicar uno u otro por lote
    },

    "entrenamiento": {
        "epocas": 20,
        "precision": "bf16",        # bf16|fp16|fp32
        "acumular": 1,              # pasos de acumulación de gradiente
        "clip_grad": 5.0,
        "congelar_epocas": 1,       # entrenar solo la cabeza al principio
        "descongelado_gradual": False,
        "paciencia": 6,             # early stopping (0 = desactivado)
        "metrica_objetivo": "acc",  # métrica que decide el mejor checkpoint
        "ema": 0.0,                 # decaimiento EMA (0.999 típico; 0 = desactivado)
        "swa": False,
        "swa_desde": 0.75,          # fracción del entrenamiento a partir de la cual promediar
        "compilar": False,          # torch.compile
        "canales_last": True,
        "tf32": True,
        "vigilar_gpu": True,        # registrar temperatura/consumo y parar si se dispara
        "temp_max": 85,             # °C; la tarjeta ya baja frecuencias sola antes de esto
        "temp_avisos": 3,           # lecturas seguidas por encima antes de parar
        "resolucion_progresiva": [],  # ej. [128, 176, 224]: sube el tamaño durante el entreno
        "reanudar": None,           # ruta a un checkpoint para continuar
    },

    # Solo para detección de anomalías visuales.
    "anomalias": {
        "tam_img": 128,
        "canales": 32,          # anchura del autocodificador
        "cuello": 64,           # tamaño del cuello de botella: cuanto más estrecho,
                                # menos puede memorizar y mejor detecta lo raro
        "percentil": 99,        # umbral: el error que solo supera el 1 % de lo normal
        "carpeta_anomalas": "", # opcional: ejemplos raros SOLO para medir, nunca entrenar
    },

    # Solo para super-resolución.
    "superresolucion": {
        "escala": 4,            # cuántas veces se agranda (2, 3, 4 u 8)
        "tam_parche": 128,      # recorte de entrenamiento en alta resolución
        "canales": 64,          # anchura de la red
        "bloques": 8,           # profundidad; más bloques = mejor y más lento
        "degradacion": 0.5,     # 0-1: cuánto se ensucia la entrada (desenfoque y JPEG)
    },

    # Solo para audio: el sonido se convierte en un espectrograma y se trata como imagen,
    # así se reaprovechan los mismos backbones y aumentaciones que en visión.
    "audio": {
        "sr": 16000,                # frecuencia de muestreo a la que se remuestrea todo
        "duracion": 3.0,            # segundos por muestra (se recorta o se rellena)
        "n_mels": 64,               # bandas del espectrograma mel
        "n_fft": 1024,
        "hop": 256,
        "specaug_tiempo": 0.15,     # SpecAugment: fracción de tiempo enmascarada
        "specaug_frec": 0.15,       # y de frecuencia
        "recorte_aleatorio": True,  # en entrenamiento, coger un trozo al azar del audio
    },

    # Solo para generación de imágenes con difusión + LoRA.
    "generacion": {
        # segmind/tiny-sd (~1 GB, OpenRAIL) va bien y es rápido.
        # Alternativas: stable-diffusion-v1-5/stable-diffusion-v1-5 (mejor calidad)
        # y stabilityai/stable-diffusion-xl-base-1.0 (la mejor, ~7 GB).
        # OJO: sdxl-turbo y sd-turbo son de uso NO comercial.
        "modelo_base": "segmind/tiny-sd",
        "resolucion": 512,
        "lora_r": 16,
        "lora_alpha": 16,
        "instancia": "una foto de sks persona",   # frase que identifica al sujeto
        "pasos": 25,                              # pasos de difusión al generar
        "guia": 7.5,                              # cuánto obedece al texto
        "negativo": "borroso, deforme, baja calidad",
    },

    # Solo para transcripción (voz a texto).
    "transcripcion": {
        "modelo_base": "openai/whisper-small",   # tiny(302MB) < base < small(2GB) < medium
        "idioma": "spanish",
        "tarea": "transcribe",                   # transcribe | translate (traduce al inglés)
        "sr": 16000,
        "duracion_max": 30.0,                    # Whisper trabaja en ventanas de 30 s
    },

    # Solo para síntesis y clonación de voz.
    "voz": {
        "modelo_base": "microsoft/speecht5_tts",       # MIT; clona por vector de hablante
        "vocoder": "microsoft/speecht5_hifigan",       # convierte el espectrograma en sonido
        "banco_voces": "Matthijs/cmu-arctic-xvectors",  # voces preajustadas (7931 vectores)
        "voz_por_defecto": 7306,                       # índice dentro del banco
        "sr": 16000,
        "tokens_max": 600,                             # límite de texto por frase
        "vector_hablante": "",                         # .npy propio; vacío = usar el banco
    },

    # Solo para datos tabulares y series temporales.
    "tabular": {
        "objetivo": "",           # columna a predecir (obligatoria)
        "ignorar": [],            # columnas a excluir (identificadores, nombres…)
        "capas": [128, 64],       # tamaño de las capas ocultas
        "dropout": 0.2,
        "tipo": "auto",           # auto | clasificacion | regresion
    },
    "series": {
        "columna": "",            # la serie a predecir
        "fecha": "",              # columna de fecha, opcional (solo para el eje)
        "ventana": 24,            # cuántos pasos mira hacia atrás
        "horizonte": 6,           # cuántos predice hacia delante
        "capas": [128, 64],
    },

    # Solo para reconocimiento de entidades (NER) y búsqueda semántica.
    "ner": {
        "modelo_base": "Davlan/bert-base-multilingual-cased-ner-hrl",  # AFL-3.0
        "longitud_max": 256,
        "etiquetas": [],          # vacío = deducirlas del dataset
    },
    "imagenes": {
        # CLIP relaciona fotos y frases en el mismo espacio. El modelo de OpenAI entiende
        # inglés; para español hay que usar un CLIP multilingüe (más pesado).
        "modelo_base": "openai/clip-vit-base-patch32",
        "lote": 32,
        "resultados": 12,
    },
    "busqueda": {
        "modelo_base": "intfloat/multilingual-e5-small",   # MIT, multilingüe
        "trozo": 250,             # caracteres por fragmento: a menor tamaño, respuestas
                                  # más precisas; a mayor, más contexto por fragmento
        "solape": 60,             # solape al partir un párrafo largo
        "resultados": 5,
    },

    # Solo para texto.
    "texto": {
        "modelo_base": "distilbert-base-multilingual-cased",
        "longitud_max": 256,
        "columna_texto": "texto",
        "columna_etiqueta": "etiqueta",
    },

    # Solo para el ajuste fino de modelos de lenguaje (LoRA / QLoRA).
    "llm": {
        "modelo_base": "Qwen/Qwen2.5-1.5B-Instruct",
        "longitud_max": 512,
        "lora_r": 16,               # rango: más alto = más capacidad y más VRAM
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_modulos": [],         # [] = deja que peft elija los del modelo
        "cuatro_bits": False,       # QLoRA: permite modelos de 7-13B en 16 GB
        "solo_respuesta": True,     # entrenar solo sobre la respuesta, no sobre el prompt
        "plantilla": "### Instrucción:\n{instruccion}\n{entrada}\n\n### Respuesta:\n",
    },

    # Solo para las tareas de rostros (identificación y verificación).
    "rostros": {
        "dim_embedding": 512,       # tamaño del vector que representa cada cara
        "min_por_identidad": 4,     # identidades con menos fotos se descartan
        "val_por_identidad": 2,     # fotos por persona reservadas para validar; con 1
                                    # no hay pares positivos y no se mide verificación
        "max_identidades": 0,       # 0 = todas
        "pares_max": 4000,          # pares que se construyen para medir verificación
        "far_objetivo": 0.001,      # TAR se reporta a esta tasa de falsa aceptación
        "umbral_similitud": 0.35,   # coseno mínimo para dar por buena una identificación
    },

    "evaluacion": {
        "tta": False,               # promediar con la imagen espejada
        "calibrar": True,           # temperature scaling sobre validación
        "curvas": True,             # AUC-ROC y precisión-recall
        "umbral": 0.5,
        "guardar_errores": True,
    },

    "salida": {
        "dir": str(RAIZ / "experimentos"),
        "tensorboard": False,
        "guardar_cada": 0,          # 0 = solo el mejor y el último
    },
}


def cargar(ruta: str | Path | None = None, preset: str | None = None,
           overrides: list[str] | None = None) -> Config:
    cfg = copy.deepcopy(DEFECTOS)
    if preset:
        _fusionar(cfg, _leer_yaml(_ruta_preset(preset)))
    if ruta:
        _fusionar(cfg, _leer_yaml(Path(ruta)))
    for override in overrides or []:
        _aplicar(cfg, override)
    return Config(cfg)


def guardar(cfg: dict, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        ruta.write_text(yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True))
    except ImportError:
        ruta.with_suffix(".json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def presets() -> list[str]:
    return sorted(p.stem for p in DIR_CONFIGS.glob("*.yaml"))


def _ruta_preset(nombre: str) -> Path:
    ruta = DIR_CONFIGS / f"{nombre}.yaml"
    if not ruta.exists():
        raise SystemExit(f"Preset '{nombre}' no encontrado. Disponibles: {', '.join(presets())}")
    return ruta


def _leer_yaml(ruta: Path) -> dict:
    if not ruta.exists():
        raise SystemExit(f"No existe la configuración {ruta}")
    try:
        import yaml
    except ImportError:
        raise SystemExit("Falta PyYAML:  pip install pyyaml") from None
    return yaml.safe_load(ruta.read_text()) or {}


def _fusionar(destino: dict, origen: dict) -> None:
    for clave, valor in origen.items():
        if isinstance(valor, dict) and isinstance(destino.get(clave), dict):
            _fusionar(destino[clave], valor)
        else:
            destino[clave] = valor


def _aplicar(cfg: dict, override: str) -> None:
    """Aplica 'datos.batch=64' o 'aumentos.mixup=0.2'."""
    if "=" not in override:
        raise SystemExit(f"Override inválido '{override}'. Formato: clave.anidada=valor")
    ruta, bruto = override.split("=", 1)
    partes = ruta.split(".")
    nodo = cfg
    for parte in partes[:-1]:
        if parte not in nodo or not isinstance(nodo[parte], dict):
            raise SystemExit(f"Sección desconocida '{parte}' en '{override}'")
        nodo = nodo[parte]
    if partes[-1] not in nodo:
        raise SystemExit(f"Opción desconocida '{ruta}'. Mira nucleo/config.py para el catálogo.")
    nodo[partes[-1]] = _convertir(bruto)


def _convertir(bruto: str):
    bajo = bruto.strip().lower()
    if bajo in ("true", "si", "sí"):
        return True
    if bajo in ("false", "no"):
        return False
    if bajo in ("none", "null", ""):
        return None
    try:
        return json.loads(bruto)          # números, listas [1,2], objetos {..}
    except json.JSONDecodeError:
        return bruto

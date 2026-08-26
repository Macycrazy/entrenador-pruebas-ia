"""Registro de vistas del panel: una por tarea, cada una con su página y su API.

Añadir una tarea al panel es añadir una entrada aquí, una plantilla y un servicio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
EXPERIMENTOS = RAIZ / "experimentos"


@dataclass(frozen=True)
class Vista:
    slug: str
    titulo: str
    icono: str
    descripcion: str
    tarea: str
    experimento: str          # carpeta por defecto en experimentos/
    entrada: str              # camara | imagen | audio | texto | chat
    config: str = ""          # config con la que se entrena esta tarea
    plantilla: str = ""       # por defecto, <slug>.html
    modelo_base: str = ""     # modelo preentrenado que la vista usa si no hay checkpoint
                              # propio: la vista funciona igual, solo que sin afinar

    @property
    def archivo_plantilla(self) -> str:
        return self.plantilla or f"{self.slug}.html"

    # Explicación en lenguaje llano, para el panel de arriba de cada página.
    # Vive en EXPLICACIONES (al final del archivo) para no ensuciar el registro.
    @property
    def explicacion(self) -> str:
        return EXPLICACIONES.get(self.slug, ("", "", ""))[0]

    @property
    def analogia(self) -> str:
        return EXPLICACIONES.get(self.slug, ("", "", ""))[1]

    @property
    def utilidad(self) -> str:
        return EXPLICACIONES.get(self.slug, ("", "", ""))[2]

    @property
    def ruta_modelo(self) -> Path:
        return EXPERIMENTOS / self.experimento / "mejor.pt"

    def entrenada(self) -> bool:
        """¿Tiene un modelo entrenado aquí?"""
        return bool(self.experimento) and self.ruta_modelo.exists()

    def disponible(self) -> bool:
        """¿Se puede usar ya? Cuenta también las que van con un modelo base."""
        if not self.experimento or self.modelo_base:
            return True
        return self.ruta_modelo.exists()

    def nivel(self) -> str:
        """entrenado | base | sin — lo que decide la insignia que ve el visitante."""
        if self.entrenada():
            return "entrenado"
        return "base" if self.disponible() else "sin"


VISTAS: list[Vista] = [
    Vista("genero", "Género por cámara", "👤",
          "Detecta rostros en la cámara y estima si la persona aparenta ser hombre o mujer.",
          "imagen_clasificacion", "genero", "camara", "configs/genero.yaml",
          "camara.html"),
    Vista("atributos", "Atributos faciales", "🎂",
          "Género, edad y etnia a la vez con un solo modelo multitarea.",
          "imagen_clasificacion", "atributos", "camara", "configs/atributos.yaml",
          "camara.html"),
    Vista("rostros", "Reconocimiento facial", "🪪",
          "Identifica quién es (1:N) o verifica si dos fotos son la misma persona (1:1).",
          "rostro_identificacion", "rostro_id", "camara", "configs/rostro_id.yaml",
          modelo_base="facenet (VGGFace2)"),
    Vista("antispoofing", "Detección de vida", "🛡️",
          "Distingue una cara real de una foto, una pantalla o una máscara.",
          "rostro_antispoofing", "antispoofing", "camara", "configs/antispoofing.yaml",
          "camara.html"),
    Vista("deteccion", "Detección de objetos", "🔲",
          "Dibuja una caja alrededor de cada objeto y dice qué es.",
          "vision_deteccion", "deteccion", "imagen", "configs/deteccion.yaml"),
    Vista("segmentacion", "Segmentación", "🎨",
          "Colorea cada píxel según la clase a la que pertenece.",
          "vision_segmentacion", "segmentacion", "imagen", "configs/segmentacion.yaml"),
    Vista("ocr", "Leer documentos", "📄",
          "Lee el texto de una foto y extrae cédula, fecha, teléfono y correo.",
          "", "", "imagen", "", "ocr.html"),
    Vista("seguimiento", "Objetos y seguimiento", "🎯",
          "Detecta 80 objetos cotidianos sin entrenar nada y los sigue entre fotogramas.",
          "", "", "camara", "", "seguimiento.html"),
    Vista("pose", "Pose corporal", "🤸",
          "Dibuja el esqueleto de cada persona: 17 articulaciones por cuerpo.",
          "", "", "camara", "", "pose.html"),
    Vista("profundidad", "Profundidad", "🌐",
          "Estima a qué distancia está cada píxel a partir de una sola foto.",
          "", "", "imagen", "", "profundidad.html"),
    Vista("anomalias", "Anomalías visuales", "⚠️",
          "Encuentra lo raro habiendo visto solo ejemplos correctos, y señala dónde está.",
          "imagen_anomalias", "anomalias", "imagen", "configs/anomalias.yaml"),
    Vista("superresolucion", "Super-resolución", "🔍",
          "Recupera detalle de una foto pequeña o borrosa, comparándose con el "
          "redimensionado normal.",
          "imagen_superresolucion", "superresolucion", "imagen",
          "configs/superresolucion.yaml"),
    Vista("audio", "Audio y voz", "🎙️",
          "Clasifica un sonido o una voz: graba con el micrófono o sube un archivo.",
          "audio_clasificacion", "voz", "audio", "configs/voz.yaml"),
    Vista("generacion", "Generar imágenes", "🎨",
          "Escribe qué quieres ver y lo dibuja. Con LoRA aprende a dibujar a una persona "
          "concreta a partir de sus fotos.",
          "imagen_generacion", "generacion", "texto", "configs/generacion.yaml",
          modelo_base="tiny-sd"),
    Vista("voz", "Hablar y clonar voz", "🗣️",
          "Escribe un texto y lo dice en voz alta; puedes clonar una voz desde una "
          "grabación de unos segundos.",
          "voz_sintesis", "voz", "texto", "configs/voz_sintesis.yaml",
          modelo_base="SpeechT5"),
    Vista("transcripcion", "Voz a texto", "✍️",
          "Graba o sube un audio y lo transcribe. Afinable a tu acento y vocabulario.",
          "audio_transcripcion", "transcripcion", "audio", "configs/transcripcion.yaml",
          modelo_base="Whisper"),
    Vista("ner", "Entidades en texto", "🏷️",
          "Encuentra nombres, lugares y organizaciones en un texto; entrenable con "
          "entidades propias.",
          "texto_ner", "ner", "texto", "configs/ner.yaml",
          modelo_base="BERT multilingüe"),
    Vista("tabular", "Predicción tabular", "📊",
          "Predice una columna a partir de las demás: ausentismo, rotación, riesgo.",
          "tabular", "tabular", "texto", "configs/tabular.yaml"),
    Vista("series", "Series temporales", "📈",
          "Predice cómo sigue una serie: asistencia, consumo, trámites por semana.",
          "series", "series", "texto", "configs/series.yaml"),
    Vista("imagenes", "Buscar fotos", "🖼️",
          "Encuentra fotos describiéndolas con palabras, sin etiquetas ni carpetas.",
          "", "", "texto", "", "imagenes.html"),
    Vista("busqueda", "Búsqueda semántica", "🔎",
          "Pregunta en lenguaje natural sobre tus documentos y encuentra el fragmento "
          "que responde.",
          "", "", "texto", "", "busqueda.html"),
    Vista("texto", "Clasificación de texto", "📝",
          "Clasifica un texto por intención, sentimiento o categoría.",
          "texto_clasificacion", "texto", "texto", "configs/texto.yaml"),
    Vista("llm", "Modelo de lenguaje", "💬",
          "Conversa con el modelo de lenguaje ajustado con LoRA.",
          "texto_llm", "llm", "chat", "configs/llm.yaml"),
]

# Páginas que no son tareas de IA: siempre disponibles.
PAGINAS: list[Vista] = [
    Vista("recorrido", "Recorrido", "🎬",
          "Cinco pasos guiados, de los datos al resultado, para enseñar el sistema.",
          "", "", "texto", "", "recorrido.html"),
    Vista("entrenamiento", "Entrenar", "🚀",
          "Lanza entrenamientos, estimaciones y descargas desde aquí, con la consola en vivo.",
          "", "", "consola", ""),
    Vista("datos", "Datos", "🗂️",
          "Con qué se entrenó: clases, reparto por subgrupo y muestras del dataset.",
          "", "", "texto", ""),
    Vista("comparar", "Comparar", "🔬",
          "La misma imagen por varios modelos, para ver qué aportó entrenar más.",
          "", "", "imagen", ""),
    Vista("guia", "Guía", "📖",
          "Cómo instalar, entrenar y usar el sistema, con los comandos exactos.",
          "", "", "texto", ""),
]

POR_SLUG = {v.slug: v for v in VISTAS + PAGINAS}

# Agrupación por familia: ordena el menú de la cabecera y la portada. Es el único
# sitio donde se decide qué tarea va en qué grupo.
_FAMILIAS = [
    ("Biometría facial", "👤",
     "Quién hay delante de la cámara, y si es una persona real.",
     ["genero", "atributos", "rostros", "antispoofing"]),
    ("Visión", "👁️",
     "Entender qué hay en una imagen y dónde está.",
     ["deteccion", "segmentacion", "seguimiento", "pose", "profundidad",
      "anomalias", "superresolucion", "ocr"]),
    ("Audio y voz", "🎙️",
     "Escuchar, transcribir y hablar.",
     ["audio", "transcripcion", "voz"]),
    ("Generación", "✨",
     "Crear contenido nuevo que no existía.",
     ["generacion"]),
    ("Texto", "📝",
     "Leer, clasificar, buscar por significado y conversar.",
     ["texto", "ner", "busqueda", "imagenes", "llm"]),
    ("Predicción", "📈",
     "Anticipar lo que va a pasar a partir de los registros.",
     ["tabular", "series"]),
]

FAMILIAS = [{"nombre": n, "icono": i, "resumen": r,
             "vistas": [POR_SLUG[s] for s in slugs if s in POR_SLUG]}
            for n, i, r, slugs in _FAMILIAS]

_agrupadas = {x.slug for f in FAMILIAS for x in f["vistas"]}
_sueltas = [x for x in VISTAS if x.slug not in _agrupadas]
if _sueltas:  # una tarea nueva sin familia asignada no desaparece del menú
    FAMILIAS.append({"nombre": "Otras", "icono": "🧩", "resumen": "", "vistas": _sueltas})


def estado() -> list[dict]:
    return [{"slug": v.slug, "titulo": v.titulo, "icono": v.icono,
             "descripcion": v.descripcion, "tarea": v.tarea, "config": v.config,
             "entrada": v.entrada, "disponible": v.disponible(),
             "nivel": v.nivel(), "modelo_base": v.modelo_base,
             "explicacion": v.explicacion, "analogia": v.analogia, "utilidad": v.utilidad,
             "experimento": v.experimento}
            for v in VISTAS]


# ---------------------------------------------------------------------------
# Qué hace cada cosa, explicado para alguien que no ha visto nunca un sistema
# de IA: (en pocas palabras, a qué se parece, para qué serviría aquí).
# ---------------------------------------------------------------------------
EXPLICACIONES: dict[str, tuple[str, str, str]] = {
    "genero": (
        "Mira la imagen de la cámara, localiza las caras y, por cada una, da un porcentaje "
        "de cuánto se parece a los rostros de hombre y de mujer que vio al entrenarse.",
        "Como alguien que ha visto cien mil fotografías y, ante una cara nueva, dice a qué "
        "grupo se parece más — sin saber nada de esa persona en concreto.",
        "Estadísticas anónimas de asistencia o aforo, sin registrar identidades."),
    "atributos": (
        "Un único modelo que aprende tres cosas a la vez. Al obligarlo a explicar género, "
        "edad y etnia con los mismos rasgos, entiende mejor la cara y acierta más en las tres.",
        "Como el empleado que además de su tarea aprende las de al lado: acaba entendiendo "
        "mejor el conjunto.",
        "Perfil demográfico agregado de quienes visitan la institución."),
    "rostros": (
        "Convierte cada cara en una lista de 512 números que la resume. Dos fotos de la misma "
        "persona dan listas parecidas; de personas distintas, listas lejanas.",
        "Como una huella dactilar, pero de la cara: no se guarda la foto, se guarda una "
        "medida de ella.",
        "Fichaje por cara, o comprobar que quien presenta el carnet es realmente su dueño."),
    "antispoofing": (
        "Comprueba que delante de la cámara hay una persona de verdad y no alguien enseñando "
        "una fotografía o la pantalla del teléfono. Se fija en la textura y los reflejos.",
        "Como el vigilante que no solo mira la foto del carnet, sino que levanta la vista "
        "para comprobar que hay alguien detrás.",
        "Impedir que alguien acceda mostrando la foto de otro al lector."),
    "deteccion": (
        "No solo dice qué hay en la imagen, sino dónde está exactamente cada cosa, "
        "rodeándola con un recuadro. Se le pueden enseñar objetos propios.",
        "Pasar de decir «aquí hay una oficina» a señalar con el dedo cada silla, cada "
        "monitor y cada persona.",
        "Comprobar equipos de protección, contar inventario o vigilar zonas restringidas."),
    "segmentacion": (
        "En vez de un recuadro, marca el contorno exacto de cada cosa, punto por punto. "
        "Sabe qué píxeles son persona, cuáles fondo y cuáles suelo.",
        "Como recortar con tijeras siguiendo el borde, en lugar de recortar un rectángulo "
        "alrededor.",
        "Quitar el fondo de las fotos de carnet automáticamente, o medir superficies."),
    "anomalias": (
        "Aprende cómo son las cosas cuando están bien y avisa cuando algo no encaja, aunque "
        "nunca haya visto ese defecto concreto. Además marca dónde está el problema.",
        "Como quien conoce su casa tan bien que nota al instante que algo está fuera de "
        "sitio, sin saber de antemano qué iba a ser.",
        "Control de calidad de carnets impresos o detección de documentos alterados."),
    "pose": (
        "Localiza hombros, codos, rodillas y demás articulaciones de cada persona y las une "
        "formando un esqueleto. Con eso se conoce la postura sin identificar a nadie.",
        "Como los muñecos de palitos que se dibujan de niño, pero calcados sobre la persona "
        "real y en movimiento.",
        "Ergonomía en puestos de trabajo, o detectar una caída en una zona sin vigilancia."),
    "profundidad": (
        "Calcula qué está cerca y qué está lejos usando una sola fotografía, sin sensores ni "
        "segunda cámara. Lo deduce de las pistas visuales, como hacemos con un ojo cerrado.",
        "Mirar una foto y saber que la persona está delante y la pared detrás, aunque la "
        "imagen sea completamente plana.",
        "Separar a la persona del fondo, o avisar si alguien se acerca demasiado a un equipo."),
    "superresolucion": (
        "Agranda una imagen reconstruyendo el detalle que falta de forma coherente, en vez "
        "de limitarse a estirar los píxeles. Aprendió viendo miles de fotos estropeadas "
        "a propósito y su versión buena.",
        "Como un restaurador que no solo amplía un cuadro dañado, sino que reconstruye las "
        "pinceladas que faltan porque sabe cómo pintaba el autor.",
        "Rescatar fotografías viejas o de mala calidad de los carnets antiguos."),
    "ocr": (
        "Encuentra dónde hay texto en la imagen, lo lee, y después reconoce automáticamente "
        "los datos con formato conocido: la cédula, una fecha, un teléfono o un correo.",
        "Como alguien que ojea un documento y va apuntando los datos importantes en una "
        "ficha, sin transcribirlo entero.",
        "Cargar los datos de una cédula al sistema con una foto, sin teclear nada."),
    "seguimiento": (
        "Reconoce ochenta tipos de objeto corriente —personas, vehículos, mobiliario— y "
        "además le asigna un número a cada uno para no perderle la pista mientras se mueve.",
        "Como el portero que no solo ve gente entrar, sino que sabe que la persona que sale "
        "es la misma que entró hace un rato.",
        "Contar cuánta gente pasó por un pasillo sin identificar a ninguna."),
    "audio": (
        "Convierte el sonido en una imagen —un mapa de frecuencias a lo largo del tiempo— y "
        "la clasifica igual que clasificaría una fotografía.",
        "Como leer la partitura de un sonido en vez de escucharlo: la forma del dibujo "
        "delata qué lo produjo.",
        "Distinguir quién habla por teléfono, o detectar una alarma dentro de una grabación."),
    "transcripcion": (
        "Escucha una grabación y escribe lo que se dijo. Funciona ya de fábrica; entrenarlo "
        "con grabaciones propias le enseña el acento local y las siglas de la casa.",
        "Una secretaria que toma dictado, y que mejora cuanto más te oye hablar.",
        "Actas de reunión automáticas, o transcribir llamadas de atención al público."),
    "voz": (
        "Convierte texto escrito en voz hablada. El timbre lo decide un «retrato sonoro» de "
        "la persona, que se extrae de unos segundos de grabación suya.",
        "Como un locutor capaz de imitar la voz de alguien tras oírle hablar un momento.",
        "Avisos por megafonía o material accesible para personas con discapacidad visual. "
        "Siempre con permiso de quien pone la voz."),
    "generacion": (
        "Crea imágenes que no existen a partir de una descripción escrita. Parte de puro "
        "ruido y lo va limpiando paso a paso hasta que se parece a lo que se pidió.",
        "Como un dibujante que, escuchando una descripción, hace un boceto y lo va refinando "
        "hasta terminarlo.",
        "Material gráfico para campañas internas sin depender de bancos de imágenes."),
    "ner": (
        "Lee un texto corrido y subraya qué palabras son nombres de persona, de lugar o de "
        "organización. Entrenándolo, reconoce también gerencias, códigos y trámites propios.",
        "Como quien lee un informe con un marcador en la mano y va resaltando cada nombre "
        "importante.",
        "Extraer automáticamente a quién y a qué se refiere cada solicitud que entra."),
    "tabular": (
        "Aprende de una tabla —la misma que sale de cualquier sistema— qué combinaciones de "
        "datos suelen terminar en un resultado, y lo estima para casos nuevos.",
        "Como el jefe con veinte años de oficio que, viendo la ficha de alguien, intuye lo "
        "que va a pasar. Pero con miles de fichas en la cabeza a la vez.",
        "Anticipar ausentismo o rotación de personal para planificar turnos."),
    "series": (
        "Mira cómo ha evolucionado un número a lo largo del tiempo, aprende sus ciclos —el "
        "fin de semana, las vacaciones— y estima los próximos valores.",
        "Como quien lleva años en recepción y sabe que los lunes de agosto viene poca gente, "
        "sin mirar ninguna estadística.",
        "Planificar personal y recursos según la asistencia prevista."),
    "imagenes": (
        "Busca imágenes escribiendo lo que quieres ver, aunque nadie las haya etiquetado ni "
        "ordenado. Entiende a la vez el contenido de la foto y el significado de la frase.",
        "Como pedirle a un archivero «tráeme las fotos donde alguien lleva gafas» y que las "
        "encuentre sin haber hecho antes ninguna lista.",
        "Encontrar material en un archivo fotográfico que nunca se catalogó."),
    "busqueda": (
        "Busca por significado y no por palabras: encuentra el párrafo que responde a la "
        "pregunta aunque no comparta ni una sola palabra con ella.",
        "Preguntarle a un compañero que se leyó todo el reglamento, en vez de usar el "
        "buscador de palabras exactas de un PDF.",
        "Consultar reglamentos y manuales internos preguntando en español normal."),
    "texto": (
        "Lee un texto y decide a qué categoría pertenece: si es una queja, una consulta o "
        "una solicitud; si el tono es positivo o negativo.",
        "Como quien reparte el correo entrante en bandejas según de qué va cada carta, sin "
        "llegar a leerlas enteras.",
        "Encaminar cada solicitud al departamento que corresponde, automáticamente."),
    "llm": (
        "Un modelo que conversa y redacta, ajustado con ejemplos propios para que responda "
        "como interesa aquí. Funciona en esta máquina, sin enviar nada a internet.",
        "Un asistente que ha leído muchísimo y al que además se le enseñan las normas de la "
        "casa para que conteste como corresponde.",
        "Redactar borradores o resumir documentos internos sin que la información salga."),
    "entrenamiento": (
        "Aquí se entrena. Se elige qué enseñar y con qué datos, y se ve en directo cómo el "
        "error baja y el acierto sube, época tras época.",
        "Como ver a alguien practicar: al principio falla mucho y, con cada repetición, se "
        "equivoca un poco menos.",
        "Lanzar y vigilar los entrenamientos sin tocar la línea de comandos."),
    "datos": (
        "Todo lo que sabe el sistema viene de estos ejemplos. Aquí se ve cuántos hay de cada "
        "tipo y, sobre todo, si están bien repartidos.",
        "El temario con el que estudió: si solo tuvo un tema, en el examen fallará todo lo "
        "demás.",
        "Comprobar que los datos representan a toda la gente y no solo a unos pocos."),
    "comparar": (
        "Pasa la misma foto por varios modelos a la vez para ver, en la práctica, qué "
        "diferencia hizo entrenar más tiempo o con más datos.",
        "El antes y el después de una reforma, puestos uno al lado del otro.",
        "Justificar con hechos que la inversión en entrenamiento sirvió para algo."),
}

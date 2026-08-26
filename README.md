# Entrenador de pruebas de IA

Un sistema para **enseñar y probar qué puede hacer una inteligencia artificial**, montado
sobre un ordenador normal. Se entrena con datos propios, funciona sin conexión a internet
y trae 23 demostraciones que se prueban desde el navegador.

---

# Parte 1 · Para quien no sabe de tecnología

Esta parte no tiene ni un comando. Es para entender qué es esto y qué se ve.

## ¿Qué es?

Es un programa que **aprende de ejemplos**.

No se le explican reglas. Se le enseñan miles de casos ya resueltos —por ejemplo, cien mil
fotografías de caras, cada una con una etiqueta— y él solo va encontrando los patrones. Al
terminar, sabe responder ante casos que nunca vio.

Todo ocurre **dentro del ordenador**. No es una suscripción a un servicio de fuera, no
manda nada por internet y no cobra por consulta.

## ¿Cómo aprende?

Siempre igual, sea para reconocer una cara o para predecir cuánta gente vendrá el mes que
viene. Cuatro pasos:

| | Paso | Qué pasa |
|---|---|---|
| **1** | **Ejemplos** | Se le dan miles de casos ya resueltos. Sin ejemplos no hay nada que aprender. |
| **2** | **Ensayo y error** | Mira un caso, responde, se le dice cuánto falló y se corrige. Millones de veces. Eso es entrenar. |
| **3** | **Examen** | Se le prueba con casos que **nunca vio**. Si solo acertara los que estudió, se los habría aprendido de memoria. |
| **4** | **En uso** | Lo aprendido queda guardado en un archivo. A partir de ahí responde en milésimas de segundo. |

## ¿Qué se puede ver funcionando?

Se abre en el navegador y cada demostración tiene su página, con una explicación arriba y
ejemplos listos para pulsar (no hace falta traer fotos ni grabaciones).

- **Caras** — reconocer quién hay delante de la cámara, y distinguir a una persona de
  verdad de alguien enseñando una fotografía.
- **Documentos** — sacarle los datos a una cédula o una constancia sin teclear nada.
- **Voz** — transcribir una reunión, y leer un texto en voz alta.
- **Imágenes** — rescatar el detalle de una foto vieja y borrosa; crear una imagen a
  partir de una descripción escrita.
- **Textos** — responder preguntas sobre nuestros propios reglamentos, buscando por
  significado y no por palabras exactas.
- **Predicción** — anticipar ausentismo o asistencia a partir de los registros.

También se puede **ver aprender al sistema en directo**: se lanza un entrenamiento desde
la propia página y se ve la línea del error bajando, minuto a minuto.

## Lo que conviene tener claro

- **Es la versión pequeña.** Modelos modestos, entrenamientos de un par de horas y una
  sola tarjeta gráfica de escritorio.
- **Se equivoca, y lo dice.** Los porcentajes están ajustados para ser honestos: cuando
  dice 90 %, acierta aproximadamente 90 de cada 100 veces.
- **Estima apariencia, no identidad.** El modelo de caras dice a qué se parece una
  fotografía según los ejemplos que vio. No determina ningún dato real de nadie.
- **Grabar rostros es tratar datos biométricos.** Hay que avisar a quien esté delante de
  la cámara.

## ¿Y a escala real?

Con un servidor dedicado la misma arquitectura da para modelos mayores, para los datos de
toda la institución y para varios entrenamientos a la vez. Lo que aquí tarda dos horas
tardaría minutos, y lo que aquí es una demostración pasaría a ser un servicio en marcha.

---

# Parte 2 · Para quien sabe de informática

Sin entrar en la maquinaria interna. Para eso está la pestaña **Guía** dentro del propio
sistema, y los comentarios del código.

## Cómo está partido

Son **dos mitades que no se pisan**:

- **El entrenador** (`entrenar.py`, `nucleo/`, `tareas/`) — se maneja por línea de
  comandos. Es el que consume la tarjeta gráfica.
- **El panel** (`servidor/`) — una web que **solo lee** los modelos ya entrenados. Nunca
  entrena, así que una cámara en vivo nunca pelea por la tarjeta con un entrenamiento.

Se pueden instalar por separado, incluso en máquinas distintas.

## Ponerlo en marcha

Hace falta Python 3.10 o más nuevo.

### Solo el panel (para verlo funcionando)

Con esto se ven las demostraciones. Vale con CPU.

```bash
git clone https://github.com/Macycrazy/entrenador-pruebas-ia.git
cd entrenador-pruebas-ia

python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r servidor/requirements.txt

python servidor/app.py --puerto 8010
```

Se abre en <http://127.0.0.1:8010>. Cinco demostraciones funcionan al momento, porque sus
modelos vienen dentro del repositorio. El resto van con modelos preentrenados que se
descargan solos la primera vez que se abre cada página (hacen falta unos gigas y
conexión).

> La cámara y el micrófono solo funcionan en `localhost` o por HTTPS. Es una norma de los
> navegadores, no del programa.

### También el entrenador (para entrenar de verdad)

Aquí sí conviene una tarjeta gráfica.

```bash
pip install -r requirements.txt
python preparacion/comprobar_gpu.py     # dice si la tarjeta está bien reconocida
```

**Si la tarjeta es una RTX 50xx**, hay que instalar PyTorch con CUDA 12.8. Las versiones
`cu118` y `cu121` no la reconocen:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Cuidado con `facenet-pytorch`**: instálalo siempre con `--no-deps`. Si no, arrastra una
versión vieja de PyTorch y deja la tarjeta sin usar.

```bash
pip install facenet-pytorch --no-deps
```

## Entrenar algo

Se puede desde la pestaña **Entrenar** del panel, con la consola en vivo, o así:

```bash
# 1. conseguir datos (o generarlos: hay un generador por tarea en preparacion/)
python preparacion/descargar_dataset.py --fuente fairface --max-por-clase 0
python preparacion/preparar_datos.py --val 0.15

# 2. saber cuánto va a tardar, antes de comprometer horas
python estimar_tiempo.py --config configs/genero.yaml

# 3. entrenar
python entrenar.py --config configs/genero.yaml
python entrenar.py --preset rapido                  # una prueba corta
python entrenar.py --reanudar experimentos/genero   # seguir tras un corte

# 4. ver qué tal quedó, con el desglose por subgrupo
python evaluar.py experimentos/genero
```

Cualquier ajuste se cambia sin tocar archivos, con `--set seccion.clave=valor`. El
catálogo completo de opciones sale con `python entrenar.py --listar-opciones`.

En cuanto existe `experimentos/<nombre>/mejor.pt`, la vista correspondiente del panel se
activa sola.

## Qué trae

16 tipos de tarea entrenable repartidos en 23 demostraciones:

| Familia | Qué hace |
|---|---|
| **Caras** | Género, edad y etnia · reconocer quién es · detectar si es una persona real |
| **Visión** | Detectar objetos · segmentar · pose · profundidad · leer documentos · anomalías · ampliar fotos |
| **Audio** | Clasificar sonidos · transcribir · hablar y clonar una voz |
| **Texto** | Clasificar · encontrar nombres y lugares · buscar por significado · conversar |
| **Datos** | Predecir una columna de una tabla · predecir cómo sigue una serie |

Cada una con su archivo en `configs/`. La tabla completa, con el estado real de cada una,
está en la pestaña **Guía → Para montarlo** del sistema.

## Qué no viene en el repositorio

Para no publicar datos de personas ni cientos de megas que se descargan solos:

| | Cómo conseguirlo |
|---|---|
| Las fotos de caras (1,5 GB) | `python preparacion/descargar_dataset.py --fuente fairface` |
| Los modelos preentrenados | Se descargan solos al abrir cada vista |
| Los índices de búsqueda | Se crean desde las propias vistas *Búsqueda semántica* y *Buscar fotos* |
| La galería de caras inscritas | `python galeria_rostros.py --inscribir <carpeta>` |
| Los datos de detección y audio | Los generan los scripts de `preparacion/` |

Sí vienen los cinco modelos ya entrenados (26 MB), para que el panel enseñe algo desde el
primer minuto.

## Cuánto tarda entrenar

Aproximado en una RTX 5060 Ti, ±40 %. Con parada temprana suele quedarse en la mitad.

| Entrenamiento | Datos | Tiempo |
|---|---|---|
| Caras, prueba rápida | 83 000 fotos, 8 vueltas | ~15 min |
| **Caras, calidad** | 83 000 fotos, 25 vueltas | **1,5 – 2 h** |
| Reconocimiento facial | 13 000 fotos | ~20 min |
| Detección de objetos | 5 000 imágenes | 40 min – 3 h |
| Audio | 20 000 clips | 10 – 20 min |
| Texto | 10 000 textos | ~5 min |
| Modelo de lenguaje con LoRA | 5 000 instrucciones | ~30 min |

## ¿Se puede estropear la tarjeta?

No. La tarjeta baja sola las revoluciones antes de sufrir daño. Y por si acaso, el
entrenamiento vigila la temperatura y **se detiene solo**, guardando lo aprendido, si pasa
de 85 °C en tres lecturas seguidas.

```bash
python preparacion/vigilar_gpu.py                   # monitor aparte
python entrenar.py --set entrenamiento.temp_max=80  # más conservador
```

Si se corta la luz no se pierde nada: cada vuelta guarda el estado completo y se retoma
con `--reanudar`.

## Un par de cosas que ahorran disgustos

- Mirar el **acierto balanceado y por subgrupo**, no el global. Con datos desequilibrados
  un modelo inútil saca 92 % global y falla todo lo que importa.
- Los porcentajes del panel están **calibrados**: no son la confianza cruda del modelo.
- El panel descarga modelos por uso (tres a la vez como mucho, ajustable con
  `PANEL_MAX_MODELOS`), porque cargarlos todos agota la memoria.

## Licencia y datos de terceros

El código es libre; ver [LICENSE](LICENSE). Los datos y los modelos preentrenados que se
descargan tienen cada uno la suya —FairFace es CC BY 4.0, y los modelos base van de MIT a
OpenRAIL— y conviene revisarla antes de darle un uso comercial.

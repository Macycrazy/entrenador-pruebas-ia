"""Tarea: detección de vida (anti-spoofing) — cara real frente a foto, pantalla o máscara.

Por dentro es clasificación de imagen, pero con un matiz importante: la pista no está
en la forma de la cara sino en la **textura** (moiré de la pantalla, reflejos del papel,
bordes del recorte, falta de microtextura de piel). Por eso hereda el comportamiento
pero conviene entrenarla con aumentaciones suaves: desenfocar, comprimir o mezclar
imágenes borra justo la señal que hay que aprender. Ver configs/antispoofing.yaml.

Dataset esperado (dos clases):

    datos_spoof/
        real/     fotos de personas delante de la cámara
        ataque/   fotos de fotos, de pantallas, máscaras impresas…
"""

from __future__ import annotations

from nucleo.tarea import registrar
from tareas.imagen_clasificacion import TareaImagenClasificacion


@registrar("rostro_antispoofing")
class TareaRostroAntispoofing(TareaImagenClasificacion):
    """Idéntica a clasificación de imagen; existe como tarea propia para que el
    checkpoint diga qué es y para poder darle sus propios valores por defecto."""

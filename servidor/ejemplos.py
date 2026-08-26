"""Ejemplos de un clic para cada vista.

El visitante que prueba el sistema no lleva encima una foto ni una grabación, así que
cada vista ofrece material listo. Se busca en varias carpetas candidatas y solo se
ofrece lo que existe de verdad en esta máquina: si el dataset no está descargado, el
botón sencillamente no aparece en vez de dar un error.

Las muestras propias viven en `servidor/ejemplos/` (van en el repositorio). El resto
salen de los datasets locales, que no se versionan.
"""

from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PROPIOS = Path(__file__).resolve().parent / "ejemplos"

EXTENSIONES = (".jpg", ".jpeg", ".png", ".webp")

# slug de la vista -> lista de (etiqueta, carpeta o archivo, cuántos coger)
# El orden importa: se sirve el primero que exista.
FUENTES: dict[str, list[tuple[str, str, int]]] = {
    # --- caras: del conjunto de validación, que nunca se usó para entrenar ---------
    "genero":       [("Rostros", "datos/val/mujer", 3), ("Rostros", "datos/val/hombre", 3)],
    "atributos":    [("Rostros", "datos/val/mujer", 3), ("Rostros", "datos/val/hombre", 3)],
    "antispoofing": [("Rostros", "datos/val/hombre", 3), ("Rostros", "datos/val/mujer", 3)],
    "rostros":      [("Rostros", "datos/val/mujer", 4), ("Rostros", "datos/val/hombre", 4)],

    # --- visión general -----------------------------------------------------------
    "ocr":          [("Documento", "servidor/ejemplos/documento.png", 1)],
    "deteccion":    [("Escena", "datos_deteccion/val/imagenes", 4),
                     ("Escena", "datos_deteccion/imagenes", 4)],
    "segmentacion": [("Escena", "datos_segmentacion/val/imagenes", 4)],
    "profundidad":  [("Documento", "servidor/ejemplos/documento.png", 1),
                     ("Rostros", "datos/val/hombre", 3)],
    "seguimiento":  [("Escena", "datos_deteccion/val/imagenes", 3),
                     ("Rostros", "datos/val/mujer", 3)],
    "pose":         [("Rostros", "datos/val/hombre", 3)],

    # --- las que ya tienen datos sintéticos propios --------------------------------
    "anomalias":    [("Correctas", "datos_anomalias/normal", 3),
                     ("Con defecto", "datos_anomalias/anomalas", 3)],
    "superresolucion": [("Rostros", "datos/val/mujer", 3),
                        ("Documento", "servidor/ejemplos/documento.png", 1)],
    "comparar":     [("Rostros", "datos/val/mujer", 2), ("Rostros", "datos/val/hombre", 2)],
}


def _archivos(rel: str, cuantos: int) -> list[Path]:
    ruta = RAIZ / rel
    if ruta.is_file():
        return [ruta]
    if not ruta.is_dir():
        return []
    # ordenados por nombre: la muestra es la misma en cada máquina, no una al azar
    todos = sorted(p for p in ruta.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTENSIONES)
    if not todos:
        return []
    # repartidos a lo largo de la carpeta, para que no salgan tres casi idénticas
    paso = max(1, len(todos) // max(cuantos, 1))
    return todos[::paso][:cuantos]


def listar(slug: str) -> list[dict]:
    """Ejemplos disponibles para una vista: [{etiqueta, url}, ...] (vacío si no hay)."""
    salida: list[dict] = []
    for etiqueta, rel, cuantos in FUENTES.get(slug, []):
        for archivo in _archivos(rel, cuantos):
            salida.append({"etiqueta": etiqueta,
                           "url": f"/api/{slug}/ejemplo/{len(salida)}"})
    return salida


def archivo(slug: str, indice: int) -> Path | None:
    """El archivo que corresponde al índice que devolvió listar()."""
    n = 0
    for _etiqueta, rel, cuantos in FUENTES.get(slug, []):
        for ruta in _archivos(rel, cuantos):
            if n == indice:
                return ruta
            n += 1
    return None

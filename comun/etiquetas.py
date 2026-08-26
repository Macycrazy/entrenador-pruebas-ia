"""Constantes compartidas: clases y normalización.

El orden de CLASES coincide con el que usa `torchvision.datasets.ImageFolder`
(alfabético: hombre=0, mujer=1). No cambiar sin reentrenar.
"""

CLASES = ["hombre", "mujer"]

# Normalización estándar de ImageNet (los pesos preentrenados la esperan).
MEDIA = (0.485, 0.456, 0.406)
DESV = (0.229, 0.224, 0.225)

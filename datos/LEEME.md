# Estructura de datos

```
datos/
├── crudo/          # imágenes originales, tal como las descargues o captures
│   ├── hombre/
│   └── mujer/
├── train/          # generado por preparar_datos.py — NO editar a mano
│   ├── hombre/
│   └── mujer/
└── val/            # generado por preparar_datos.py
    ├── hombre/
    └── mujer/
```

Los nombres `hombre` y `mujer` son obligatorios y su orden alfabético define los índices
de clase del modelo (hombre = 0, mujer = 1). Ver `comun/etiquetas.py`.

Flujo:

```bash
# 1. conseguir imágenes en crudo/hombre y crudo/mujer
python entrenamiento/descargar_dataset.py --fuente fairface --max-por-clase 0   # de internet
python entrenamiento/importar_utkface.py --origen ~/Descargas/UTKFace          # UTKFace local
#    …o copiarlas a mano en crudo/hombre y crudo/mujer

# 2. recortar rostros y repartir en train/val
python entrenamiento/preparar_datos.py --recortar --val 0.15 --limpiar
```

`.cache_parquet/` es temporal: lo crea y lo borra `descargar_dataset.py`.

Recomendaciones: mínimo ~2 000 imágenes por clase, variedad de edad, etnia, iluminación
y ángulo, y ninguna persona repetida entre `train` y `val` (si no, la precisión de validación
sale inflada).

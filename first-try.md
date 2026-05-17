# Registro del primer notebook: VISEM-Tracking EDA + YOLOv8 baseline

Este documento resume lo realizado en el primer notebook `initial-visem-tracking-eda-yolo.ipynb`, los resultados obtenidos y la razón por la que se decidió crear una versión mejorada.

## 1. Objetivo del primer notebook

El primer notebook tuvo como objetivo construir una primera tubería completa para trabajar con el dataset **VISEM-Tracking** usando YOLOv8. La idea principal era validar que el dataset se pudiera descargar, leer, convertir a formato YOLO, entrenar y evaluar correctamente.

La tubería realizada fue:

1. Instalar dependencias necesarias.
2. Descargar el dataset VISEM-Tracking desde Zenodo.
3. Descomprimir el dataset en Kaggle.
4. Inspeccionar la estructura de carpetas.
5. Buscar imágenes y archivos de anotación `.txt`.
6. Asociar cada imagen con su label YOLO correspondiente.
7. Leer las anotaciones.
8. Realizar EDA básica.
9. Visualizar ejemplos con bounding boxes.
10. Hacer split por video.
11. Crear el dataset en formato YOLO.
12. Entrenar un baseline con YOLOv8n.
13. Evaluar en test.
14. Generar predicciones visuales.
15. Ejecutar tracking con ByteTrack sobre un video.

## 2. Descarga y estructura del dataset

El notebook descargó correctamente el dataset desde Zenodo. El ZIP descargado tuvo un tamaño aproximado de **5.88 GB**.

Después de descomprimir, se encontró la carpeta principal:

```text
VISEM_Tracking_Train_v4
```

Dentro de esta carpeta se encontraron videos, imágenes extraídas y anotaciones YOLO organizadas por video.

También se encontraron carpetas adicionales con clips de video extraídos, además de archivos `.csv` relacionados con información clínica o de análisis de semen. Para este primer notebook se usaron principalmente las imágenes, labels y videos necesarios para detección/tracking.

## 3. Imágenes y labels encontrados

El notebook encontró:

- **29,370 imágenes**.
- **58,412 archivos `.txt`**.
- **29,196 pares imagen-label** válidos.

Esto indica que casi todas las imágenes relevantes tenían un archivo de anotación asociado. La diferencia entre imágenes totales y pares encontrados no fue problemática para el entrenamiento, porque se trabajó solo con pares válidos.

## 4. Detalle corregido sobre `video_id`

En una parte inicial del notebook, el campo `video_id` quedó asignado incorrectamente como:

```text
VISEM_Tracking_Train_v4
```

Esto ocurrió porque inicialmente se estaba tomando una carpeta demasiado general de la ruta.

Más adelante, el notebook corrigió este problema usando una función para inferir mejor el `video_id` desde la ruta real de cada imagen. Después de esa corrección se detectaron correctamente **20 videos**:

```text
11, 12, 13, 14, 15, 19, 21, 22, 23, 24, 29, 30, 35, 36, 38, 47, 52, 54, 60, 82
```

Este punto fue importante porque el split debía hacerse por video y no por frame.

## 5. Anotaciones leídas

El notebook leyó correctamente las anotaciones YOLO. En total se obtuvieron:

- **656,335 anotaciones**.

Las clases usadas fueron:

| ID | Clase |
|---:|---|
| 0 | sperm |
| 1 | cluster |
| 2 | small_or_pinhead |

## 6. Distribución de clases

El EDA mostró un desbalance fuerte entre clases:

| Clase | Número de anotaciones |
|---|---:|
| sperm | 612,377 |
| cluster | 21,846 |
| small_or_pinhead | 22,112 |

La clase `sperm` dominó claramente el dataset. Esto fue una señal importante, porque un modelo entrenado sin balanceo tendería a aprender principalmente la clase mayoritaria.

## 7. Objetos por imagen

El análisis de objetos por frame mostró:

| Estadístico | Valor |
|---|---:|
| count | 29,196 |
| mean | 22.48 |
| std | 16.60 |
| min | 1 |
| 25% | 9 |
| 50% | 21 |
| 75% | 34 |
| max | 71 |

Esto confirmó que muchas imágenes tienen bastantes objetos. Por eso, para los experimentos mejorados conviene usar `max_det=1000`, evitando limitar artificialmente la cantidad de detecciones por imagen.

## 8. Tamaño de las bounding boxes

El área normalizada de las cajas tuvo estos valores:

| Estadístico | Valor |
|---|---:|
| count | 656,335 |
| mean | 0.001959 |
| std | 0.002085 |
| min | 0.000052 |
| 25% | 0.001088 |
| 50% | 0.001504 |
| 75% | 0.002187 |
| max | 0.029997 |

Esto mostró que muchos objetos son pequeños. Por esa razón, entrenar con `imgsz=640` puede quedarse corto. Para mejorar el modelo se decidió subir la resolución a `imgsz=960` o incluso `1280` si la GPU lo permite.

## 9. Split por video

El notebook hizo correctamente el split por video, evitando fuga de información entre frames del mismo video.

La distribución final fue:

| Split | Imágenes |
|---|---:|
| train | 20,550 |
| val | 4,236 |
| test | 4,410 |

Los videos quedaron distribuidos así:

```text
train: 13, 14, 15, 21, 22, 24, 29, 30, 35, 36, 38, 52, 60, 82
val:   12, 23, 47
test:  11, 19, 54
```

Este diseño fue correcto y se mantiene en el notebook mejorado.

## 10. Dataset YOLO creado

El primer notebook creó correctamente la estructura:

```text
/kaggle/working/yolo_visem/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```

La cantidad de imágenes y labels quedó consistente:

| Split | Imágenes | Labels |
|---|---:|---:|
| train | 20,550 | 20,550 |
| val | 4,236 | 4,236 |
| test | 4,410 | 4,410 |

## 11. Entrenamiento baseline

Se entrenó un primer baseline con:

```python
model = YOLO("yolov8n.pt")

results = model.train(
    data=str(YAML_PATH),
    epochs=30,
    imgsz=640,
    batch=16,
    patience=8,
    project=str(WORK_DIR / "runs"),
    name="visem_yolov8n",
    pretrained=True
)
```

Este modelo fue útil como prueba inicial porque es liviano y permitió validar toda la tubería. Sin embargo, era esperable que tuviera limitaciones por tres razones:

1. `yolov8n` es la versión más pequeña.
2. `imgsz=640` puede ser bajo para objetos pequeños.
3. El dataset está fuertemente desbalanceado hacia la clase `sperm`.

## 12. Resultados en test del baseline

El baseline obtuvo estos resultados en test:

| Clase | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.195 | 0.193 | 0.150 | 0.0419 |
| sperm | 0.561 | 0.578 | 0.429 | 0.117 |
| cluster | 0.0247 | 0.000347 | 0.021 | 0.00834 |
| small_or_pinhead | 0 | 0 | 0.00000195 | 0.000000195 |

La lectura principal es que el modelo sí aprendió parcialmente a detectar `sperm`, pero prácticamente no aprendió las clases minoritarias `cluster` y `small_or_pinhead`.

## 13. Predicciones visuales

El notebook generó predicciones sobre imágenes de test. La mayoría de detecciones fueron de la clase `sperm`. En algunos ejemplos apareció `small_or_pinhead`, pero las métricas globales mostraron que este comportamiento no fue consistente.

Esto confirmó que el modelo funcionaba como baseline visual, pero no era suficiente como modelo final.

## 14. Tracking con ByteTrack

También se probó tracking con ByteTrack sobre un video original. El video usado tenía **1,470 frames**.

El tracking corrió, pero apareció una advertencia de Ultralytics indicando que los resultados se acumularían en RAM si no se usaba `stream=True`.

Por eso, en el notebook mejorado se usa:

```python
stream=True
```

junto con:

```python
for _ in tracking_results:
    pass
```

Esto evita consumo innecesario de memoria en videos largos.

## 15. Conclusión del primer notebook

El primer notebook fue exitoso como baseline porque demostró que:

- El dataset se podía descargar y procesar en Kaggle.
- Las imágenes y labels se podían asociar correctamente.
- Las anotaciones estaban en formato YOLO válido.
- El split por video era posible.
- El dataset se podía convertir a estructura YOLO.
- YOLOv8 podía entrenarse y evaluarse.
- ByteTrack podía correr sobre los videos originales.

Sin embargo, los resultados mostraron que el modelo baseline no era suficiente para entregar el mejor trabajo posible.

## 16. Razones para crear el notebook mejorado

A partir de los resultados del baseline, se decidió mejorar el enfoque con estos cambios:

1. **Corregir el manejo de `video_id` desde el inicio**, no después.
2. **Mantener split por video** para evitar fuga de información.
3. **Subir resolución** de `640` a `960`, porque los objetos son pequeños.
4. **Usar un modelo más fuerte**, empezando con `yolov8s.pt` y dejando opción de `yolov8m.pt`.
5. **Sobremuestrear imágenes con clases minoritarias** en train.
6. **Mantener val y test sin modificar** para evaluación justa.
7. **Aumentar épocas y paciencia** para permitir mejor convergencia.
8. **Usar `max_det=1000`** porque hay frames con muchos objetos.
9. **Usar `stream=True` en tracking** para evitar problemas de RAM.
10. **Guardar resultados en un ZIP final** para facilitar la entrega.

## 17. Estado del trabajo después del primer notebook

El primer notebook no se debe considerar un fracaso. Al contrario, fue una etapa exploratoria necesaria. Permitió descubrir los principales retos del problema:

- Desbalance extremo de clases.
- Objetos pequeños.
- Alta densidad de objetos por frame.
- Necesidad de dividir por video.
- Necesidad de ajustar tracking para videos largos.

El notebook mejorado parte de estos hallazgos y transforma el trabajo de un baseline simple a un experimento más sólido y defendible.

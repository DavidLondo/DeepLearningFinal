# README.md

## Información General

| Campo       | Información |
| ----------- | ----------- |
| Proyecto    | Predicción Salud Seminal |
| Curso       | Redes Neuronales y Deep Learning |
| Integrantes | Daniel Santana, Felipe Castro, David Londoño |
| Profesor    | Raúl Castañeda |
| Fecha       | 20/5/2026   |
| Documento Informe Final | [INFORME_PROYECTO - Equipo #2](https://eafit-my.sharepoint.com/:b:/g/personal/fcastroj_eafit_edu_co/IQAIK8POVQGmSLhVXKoQgNZMAd3TBKSSc83l4Iw88ILWXIM?e=bFl9mu) | 

---

# 1. Descripción General

## Contexto

La evaluación de la calidad seminal es un proceso fundamental dentro de los estudios de fertilidad masculina. Tradicionalmente, este análisis se basa en parámetros establecidos por la Organización Mundial de la Salud (OMS), incluyendo concentración espermática, movilidad, morfología y vitalidad. Sin embargo, el análisis manual puede resultar costoso, subjetivo y dependiente de especialistas.

En este proyecto se desarrolló un sistema de clasificación multimodal basado en técnicas de Machine Learning y Deep Learning para predecir anomalías seminales utilizando información clínica y características visuales extraídas automáticamente de videos microscópicos.

## Motivación

La motivación principal del proyecto consiste en explorar el potencial de modelos multimodales para asistir en el diagnóstico automatizado de alteraciones seminales. La integración de datos clínicos con información cinemática derivada del movimiento espermático puede mejorar significativamente la capacidad predictiva frente a modelos unimodales.

## Objetivo Principal

Desarrollar y evaluar modelos multimodales capaces de identificar anomalías seminales utilizando información clínica y características visuales extraídas automáticamente del dataset VISEM.

## Alcance

El proyecto contempla:

- Preprocesamiento y limpieza de datos clínicos y visuales.
- Extracción y consolidación de características multimodales.
- Implementación de modelos clásicos y modelos Deep Learning.
- Evaluación robusta mediante validación cruzada repetida.
- Comparación entre aproximaciones tradicionales y neuronales.

No se incluye despliegue clínico ni validación médica real.

---

# 2. Objetivos

## Objetivo General

Desarrollar un sistema multimodal basado en técnicas de Machine Learning y Deep Learning para la detección automática de anomalías seminales utilizando datos clínicos y características visuales extraídas de videos microscópicos.

## Objetivos Específicos

* Integrar características clínicas y visuales provenientes del dataset VISEM.
* Implementar modelos clásicos y modelos basados en Deep Learning para clasificación binaria.
* Evaluar el rendimiento de cada modelo utilizando métricas robustas al desbalanceo.
* Analizar el impacto de la multimodalidad sobre el desempeño predictivo.
* Comparar modelos tradicionales frente a arquitecturas neuronales profundas.

---

# 3. Arquitectura del Proyecto

## Estructura de Carpetas

```txt
project/
│
├── data/ 
├── notebooks/
├── models/
└── README.md
```

## Arquitectura del Modelado

 1. Arquitectura Real
  El modelo principal desarrollado en PyTorch es un Multimodal MLP diseñado para procesar dos tipos de fuentes de datos de
  forma paralela antes de fusionarlas:

   * Ramas de entrada (Feature Extractors):
       * Rama Clínica: Procesa las 51 características tabulares (edad, IMC, hormonas, ácidos grasos). Estructura:
         Linear(51, 32) → BatchNorm1d → ReLU → Dropout(0.30).
       * Rama de Video: Procesa las 19 características extraídas del movimiento. Estructura: Linear(19, 16) → BatchNorm1d
         → ReLU → Dropout(0.30).
   * Fusión: Las representaciones de ambas ramas se concatenan (fused vector de tamaño 48).
   * Clasificador Final: Linear(48, 32) → BatchNorm1d → ReLU → Dropout(0.30) → Linear(32, 1).
   * Función de Pérdida: BCEWithLogitsLoss con pos_weight calculado dinámicamente para manejar el desbalance de clases.

---

# 4. Dataset

## Fuente de Datos

Se utilizó el VISEM Dataset, un dataset multimodal diseñado para el análisis de calidad seminal humana.

## Características del Dataset

| Característica     | Valor |
| ------------------ | ----- |
| Número de participantes | 85      |
| Número total de características   |   70    |
| Características Clínicas             |  51      |
| Características Visuales            |    19   |
| Tipo de Problema            |    Clasificación Binaria   |

### Variables Clínicas
Las variables clínicas incluyen:

- Edad
- Índice de masa corporal (IMC)
- Días de abstinencia
- Hormonas sexuales
- Ácidos grasos séricos y espermáticos

### Variables Visuales
Las características visuales fueron extraídas automáticamente a partir de videos microscópicos mediante algoritmos de tracking y análisis cinemático.
Entre las variables más relevantes se encuentran:

- Velocidad media
- Desplazamiento
- Rectitud del movimiento
- Área del espermatozoide
- Parámetros cinemáticos de trayectoria

### Variable Objetivo
Se utilizó el target binario:

```txt
target_any_semen_abnormality
```

Esta variable indica la presencia de al menos una anomalía seminal según criterios de la OMS.

## División de Datos

La evaluación se realizó mediante validación cruzada repetida:

- 5 folds estratificados
- 10 semillas aleatorias
Total: 50 evaluaciones por modelo

## Preprocesamiento

Las etapas principales de preprocesamiento fueron:

1. Eliminación de columnas duplicadas con sufijos _x y _y.
2. Normalización de nombres de columnas.
3. Imputación de valores nulos mediante mediana (SimpleImputer).
4. Escalamiento de características.
5. Estratificación de folds para preservar distribución de clases.

---

# 5. Implementación

## Arquitectura Multimodal MLP

El modelo principal desarrollado en PyTorch corresponde a una arquitectura multimodal tipo MLP con dos ramas independientes

### Rama Clínica

Procesa las 51 características tabulares clínicas.
Arquitectura:

```txt
Linear(51, 32)
BatchNorm1d
ReLU
Dropout(0.30)
```

### Rama Visual

Procesa las 19 características cinemáticas extraídas de videos.
Arquitectura:

```txt
Linear(19, 16)
BatchNorm1d
ReLU
Dropout(0.30)
```

### Fusión Multimodal

Las representaciones latentes de ambas ramas son concatenadas formando un vector de tamaño 48.

### Clasificador Final

```text
Linear(48, 32)
BatchNorm1d
ReLU
Dropout(0.30)
Linear(32, 1)
```

## Función de Pérdida
Se utilizó:

```text
BCEWithLogitsLoss
```

Con ```pos_weight``` calculado dinámicamente para compensar el desbalance de clases.

## Modelos Implementados
Modelos Clásicos:
- SVM con kernel RBF
- Random Forest
- Gradient Boosting
- Logistic Regression
- MLP de Scikit-learn

## Configuración del Entrenamiento

| Parámetro     | Valor |
| ------------- | ----- |
| Learning Rate | 0.001      |
| Batch Size    | 16      |
| Max Epochs        | 300      |
| Optimizer     | AdamW      |
| Weight Decay | 0.001      |
| Dropout       | 0.30 |
| Early Stopping | Esperade 30 Epocas |

# 6. Resultados

Debido al desbalanceo de clases presente en el dataset, se utilizaron métricas robustas:
- ROC AUC
- Balanced Accuracy
- F1-Score
- Average Precision

### Mejor Modelo — SVM RBF Multimodal

| Métrica    | Resultado |
| ---------- | --------- |
| ROC AUC   |  0.879         |
| Balanced Accuracy  | 0.779          |
| F1-Score  | 0.850 |

### Modelo PyTorch — Multimodal MLP

| Métrica    | Resultado |
| ---------- | --------- |
| ROC AUC   |  0.812         |
| Balanced Accuracy  |  0.716          |
| F1-Score  | 0.757 |

### Observaciones Generales

Observaciones Generales
- La integración multimodal mejoró consistentemente el rendimiento frente a modelos unimodales.
- Las características visuales demostraron una alta capacidad predictiva incluso de forma independiente.
- El modelo SVM con kernel RBF obtuvo la mejor generalización debido al tamaño reducido del dataset.
- El modelo MLP multimodal mostró resultados competitivos y mayor potencial de escalabilidad.

# 7. Análisis de Resultados

Los resultados obtenidos evidencian que la multimodalidad representa una estrategia efectiva para la clasificación de anomalías seminales.

El modelo SVM con kernel RBF alcanzó el mejor desempeño general, obteniendo un ROC AUC de 0.879. Este comportamiento puede explicarse por la capacidad de los modelos basados en kernels para adaptarse mejor a datasets pequeños y de alta dimensionalidad.

Por otro lado, el modelo Multimodal MLP implementado en PyTorch logró un desempeño competitivo con ROC AUC de 0.812, demostrando capacidad para aprender representaciones conjuntas entre información clínica y visual.

El uso de características visuales derivadas de videos resultó especialmente relevante, ya que los experimentos utilizando únicamente variables visuales alcanzaron valores elevados de discriminación.

También se observó que el tamaño limitado del dataset favorece el overfitting en arquitecturas profundas, incluso utilizando técnicas de regularización como dropout, batch normalization y early stopping.

---

# 8. Trabajo Futuro

Posibles líneas de mejora futuras:

- Incrementar el tamaño del dataset.
- Implementar arquitecturas multimodales más profundas.
- Incorporar embeddings temporales directamente desde video.
- Utilizar modelos basados en Transformers.
- Aplicar técnicas avanzadas de data augmentation.
- Realizar validación clínica externa.
- Implementar sistemas de inferencia en tiempo real.

---

# 9. Conclusiones

El proyecto permitió desarrollar y evaluar exitosamente múltiples modelos de clasificación multimodal para la detección automática de anomalías seminales utilizando el dataset VISEM.

Los resultados demostraron que la combinación de información clínica y características visuales mejora significativamente el desempeño predictivo respecto a enfoques unimodales.

El modelo SVM con kernel RBF alcanzó el mejor rendimiento global, lo que evidencia que modelos clásicos continúan siendo altamente competitivos en escenarios con datasets pequeños.

Por otro lado, la arquitectura Multimodal MLP implementada en PyTorch mostró resultados sólidos y un importante potencial de escalabilidad para futuros escenarios con mayor volumen de datos.

Finalmente, el proyecto confirma el valor de la multimodalidad y del análisis automatizado asistido por inteligencia artificial como herramientas prometedoras para aplicaciones biomédicas relacionadas con fertilidad masculina.

---

# 10. Referencias

## Papers

* [Referencia 1]
* [Referencia 2]


## Dataset

* [Fuente del dataset]

---

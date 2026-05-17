## Conclusión del primer experimento de tracking

Se implementó un primer flujo completo de tracking utilizando el detector YOLO entrenado y el algoritmo ByteTrack. El modelo fue evaluado sobre el video 11 del conjunto de test, el cual no fue utilizado durante el entrenamiento.

Los resultados muestran que el detector logra encontrar una cantidad considerable de espermatozoides, obteniendo una precisión aproximada de 0.749, un recall aproximado de 0.929 y un F1 aproximado de 0.829 con un umbral IoU de 0.3. Esto indica que el modelo tiene una capacidad razonable para detectar objetos en frames no vistos.

Sin embargo, al analizar las trayectorias, se observa una diferencia importante entre el ground truth y las predicciones. El ground truth contiene 55 trayectorias, mientras que el sistema YOLO + ByteTrack generó 276 trayectorias. Además, la mediana de duración de las trayectorias reales es de 1283 frames, mientras que en las trayectorias predichas es de solo 23 frames. Esto evidencia una alta fragmentación de identidades, donde el tracker pierde continuidad y crea nuevos IDs para objetos que probablemente corresponden al mismo espermatozoide.

Por lo tanto, el flujo completo funciona correctamente como baseline inicial, pero la calidad del tracking todavía es mejorable. El siguiente paso debe enfocarse en reducir la fragmentación de trayectorias, ajustando los parámetros del tracker, probando el tracking directamente sobre el video original y comparando ByteTrack con otros trackers como BoT-SORT.
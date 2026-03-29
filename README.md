Aquí tienes una versión más sobria y directa para tu `README.md`:

---

# Tarea 3: Clasificación KNN y Visualización de Datos

Implementación manual del algoritmo K-Nearest Neighbors (KNN) para la clasificación de diabetes utilizando el dataset `diabetes.csv`. El proyecto incluye análisis estadístico, preprocesamiento de datos y visualización de características.

## Requisitos e Instalación

Es necesario contar con Python 3.x y las siguientes librerías:

```bash
pip install pandas numpy matplotlib seaborn
```

## Estructura del Proyecto

El código se divide en archivos independientes para cada etapa del análisis:

1. **Visualización de Datos (`visualizacion.py`)**
   * Carga y estandarización de datos (Z-score).
   * Generación de gráficas de dispersión para analizar el solapamiento de clases entre variables clave (Glucosa, IMC, Edad, etc.).

2. **Modelo KNN Manual (`knn_diabetes.py`)**
   * Implementación desde cero de métricas de distancia: Euclidiana, Manhattan, Minkowski ($p=3$) y Chebyshev.
   * Evaluación de múltiples valores de $k$ ($3, 5, 7, 11, 13, 15$).
   * Generación de matrices de confusión y reporte de *accuracy* por cada configuración.

## Ejecución

Asegúrate de que el archivo `diabetes.csv` se encuentre en el mismo directorio que los scripts:

```bash
python visualizacion.py
python knn_diabetes.py
```

## Detalles Técnicos
* **Preprocesamiento:** Los datos fueron normalizados para evitar sesgos por escalas.
* **Muestreo:** Se aplicó una división de 70% entrenamiento y 30% prueba con balanceo de clases.
* **Optimización:** Uso de operaciones vectorizadas con NumPy para el cálculo eficiente de distancias.

---
**Autor:** Juan Pablo Vera Morales
**Facultad de Ingeniería, UNAM.**

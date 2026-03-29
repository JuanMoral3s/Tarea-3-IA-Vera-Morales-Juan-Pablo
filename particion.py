import pandas as pd
import numpy as np

datos = pd.read_csv("diabetes.csv")
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

datos_std = (datos[cols] - datos[cols].mean()) / datos[cols].std()
datos_std['Outcome'] = datos['Outcome']
total_muestras = len(datos_std)
tamano_test = int(total_muestras * 0.30)
tamano_por_clase = tamano_test // 2  # 15% sanos, 15% diabéticos

# Separar y muestrear
df_neg = datos_std[datos_std['Outcome'] == 0]
df_pos = datos_std[datos_std['Outcome'] == 1]

test_neg = df_neg.sample(n=tamano_por_clase, random_state=42)
test_pos = df_pos.sample(n=tamano_por_clase, random_state=42)

test_set = pd.concat([test_neg, test_pos])
train_set = datos_std.drop(test_set.index)

# Convertir a matrices de Numpy para hacer los cálculos matemáticos más rápidos
X_train = train_set.drop('Outcome', axis=1).values
y_train = train_set['Outcome'].values
X_test = test_set.drop('Outcome', axis=1).values
y_test = test_set['Outcome'].values

# ==========================================
# FUNCIONES MATEMÁTICAS: Distancias y Métricas
# ==========================================
def calcular_distancias(X_train, x_punto, metrica, p=3):
    # Usamos broadcasting de Numpy para restar el punto de prueba a toda la matriz de entrenamiento
    diferencia = X_train - x_punto
    
    if metrica == 'euclidiana':
        return np.sqrt(np.sum(diferencia**2, axis=1))
    elif metrica == 'manhattan':
        return np.sum(np.abs(diferencia), axis=1)
    elif metrica == 'minkowski':
        return np.sum(np.abs(diferencia)**p, axis=1)**(1/p)
    elif metrica == 'chebyshev':
        return np.max(np.abs(diferencia), axis=1)

def evaluar_predicciones(y_real, y_pred):
    # Cálculo manual de Matriz de Confusión
    TP = np.sum((y_real == 1) & (y_pred == 1))
    TN = np.sum((y_real == 0) & (y_pred == 0))
    FP = np.sum((y_real == 0) & (y_pred == 1))
    FN = np.sum((y_real == 1) & (y_pred == 0))
    
    accuracy = (TP + TN) / len(y_real)
    
    # Formato de tabla para la consola
    matriz = pd.DataFrame({
        'Pred: No Diab': [TN, FN],
        'Pred: Diab': [FP, TP]
    }, index=['Real: No Diab', 'Real: Diab'])
    
    return accuracy, matriz

# ==========================================
# ALGORITMO KNN MANUAL
# ==========================================
def knn_manual(X_train, y_train, X_test, k, metrica, p=3):
    predicciones = []
    
    for x_punto in X_test:
        # 1. Calcular distancias contra todos los puntos de entrenamiento
        distancias = calcular_distancias(X_train, x_punto, metrica, p)
        
        # 2. Ordenar y obtener los índices de los 'k' más cercanos
        indices_k_cercanos = np.argsort(distancias)[:k]
        
        # 3. Ver qué etiquetas (0 o 1) tienen esos vecinos
        etiquetas_vecinos = y_train[indices_k_cercanos]
        
        # 4. Votación mayoritaria (el valor que más se repite)
        voto_ganador = np.bincount(etiquetas_vecinos).argmax()
        predicciones.append(voto_ganador)
        
    return np.array(predicciones)

# ==========================================
# 4.3 y 4.4 EXPERIMENTOS
# ==========================================
k_valores = [3, 5, 7, 11, 13, 15]
distancias_a_probar = ['euclidiana', 'manhattan', 'minkowski', 'chebyshev']

resultados = []

for dist in distancias_a_probar:
    print(f"\n{'='*40}\n--- Evaluando Distancia: {dist.upper()} ---\n{'='*40}")
    
    for k in k_valores:
        # Hacemos la predicción
        y_pred = knn_manual(X_train, y_train, X_test, k=k, metrica=dist, p=3)
        
        # Evaluamos
        acc, matriz = evaluar_predicciones(y_test, y_pred)
        resultados.append({'Distancia': dist, 'K': k, 'Accuracy': acc})
        
        print(f"\n[ K = {k} | Accuracy = {acc:.4f} ]")
        print(matriz)

# Mostrar Resumen de los mejores resultados
df_resultados = pd.DataFrame(resultados)
idx_mejores = df_resultados.groupby('Distancia')['Accuracy'].idxmax()
tabla_resumen = df_resultados.loc[idx_mejores]

print("\n\n=== TABLA RESUMEN FINAL (Mejor K por Distancia) ===")
print(tabla_resumen.to_string(index=False))
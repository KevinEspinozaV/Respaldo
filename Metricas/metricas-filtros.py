import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import sys

def matriz_to_list(matriz_confusion):

    # Obtener la dimensión de la matriz de confusión
    n_clases = len(matriz_confusion)

    # Inicializar las listas de valores reales y valores predecidos
    valores_reales = []
    valores_predecidos = []

    # Recorrer la matriz de confusión y agregar los valores a las listas
    for fila in range(n_clases):
        for columna in range(n_clases):
            valor = matriz_confusion[fila, columna]
            valores_reales.extend([fila] * valor)
            valores_predecidos.extend([columna] * valor)

    return valores_reales, valores_predecidos

# Definir las matrices
matriz1_area = np.array([[1,2,0,0],[0,3,0,0],[0,1,2,0],[0,0,0,3]]) # np.array([[2,1,0,0],[0,3,0,0],[0,1,2,0],[0,0,0,3]]) 
matriz2_area = np.array([[0,3,0,0],[0,3,0,0],[0,1,1,1],[0,0,0,3]])
matriz3_area = np.array([[3,0,0,0],[2,1,0,0],[1,1,1,0],[0,0,0,3]])
matriz4_area = np.array([[3,0,0,0],[1,2,0,0],[1,1,1,0],[0,0,1,2]])
matriz5_area = np.array([[2,1,0,0],[1,2,0,0],[1,0,2,0],[1,0,0,2]])

# Definir las matrices
matriz1_tincion = np.array([[1,2,0,0],[1,2,0,0],[0,1,1,1],[0,0,0,3]])
matriz2_tincion = np.array([[2,1,0,0],[0,2,0,1],[0,1,2,0],[0,0,0,3]])
matriz3_tincion = np.array([[1,2,0,0],[1,1,0,1],[1,0,2,0],[0,0,0,3]])
matriz4_tincion = np.array([[3,0,0,0],[2,1,0,0],[1,1,0,1],[0,0,1,2]])
matriz5_tincion = np.array([[3,0,0,0],[0,2,1,0],[0,0,3,0],[1,0,0,2]])

# Definir las matrices
matriz1_hist = np.array([[0,0,2,1],[0,1,1,1],[0,2,1,0],[0,0,0,3]])
matriz2_hist = np.array([[3,0,0,0],[1,2,0,0],[0,3,0,0],[0,0,0,3]])
matriz3_hist = np.array([[3,0,0,0],[2,1,0,0],[0,0,3,0],[0,0,0,3]])
matriz4_hist = np.array([[3,0,0,0],[1,2,0,0],[1,2,0,0],[0,0,0,3]])
matriz5_hist = np.array([[2,1,0,0],[1,1,1,0],[0,0,3,0],[0,0,0,3]])

# Definir las matrices (Estas son las matrices sumadas ya)
matriz_512 = np.array([[9,4,0,2],[3,11,1,0],[0,7,8,0],[0,0,1,14]])
matriz_256 = np.array([[9,4,2,0],[5,6,4,0],[0,6,9,0],[0,0,2,13]])

# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

# Sumar las matrices
resultado_area = matriz1_area + matriz2_area + matriz3_area + matriz4_area + matriz5_area
resultado_tincion = matriz1_tincion + matriz2_tincion + matriz3_tincion + matriz4_tincion + matriz5_tincion
resultado_hist = matriz1_hist + matriz2_hist + matriz3_hist + matriz4_hist + matriz5_hist

print(resultado_area)
print(resultado_tincion)
print(resultado_hist)
#print(matriz_512)
#print(matriz_256)

# Obtener las etiquetas reales y las predicciones
y_true_512, y_pred_512 = matriz_to_list(matriz_512)

# Obtener las etiquetas reales y las predicciones
y_true_256, y_pred_256 = matriz_to_list(matriz_256)

# Obtener las etiquetas reales y las predicciones
y_true_area, y_pred_area = matriz_to_list(resultado_area)

# Obtener las etiquetas reales y las predicciones
y_true_tincion, y_pred_tincion = matriz_to_list(resultado_tincion)

# Obtener las etiquetas reales y las predicciones
y_true_hist, y_pred_hist = matriz_to_list(resultado_hist)

# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

# Calcular las métricas
accuracy_512 = accuracy_score(y_true_512, y_pred_512)
precision_512 = precision_score(y_true_512, y_pred_512, average='weighted')
recall_512 = recall_score(y_true_512, y_pred_512, average='weighted')
f1_512 = f1_score(y_true_512, y_pred_512, average='weighted')

# Calcular las métricas
accuracy_256 = accuracy_score(y_true_256, y_pred_256)
precision_256 = precision_score(y_true_256, y_pred_256, average='weighted')
recall_256 = recall_score(y_true_256, y_pred_256, average='weighted')
f1_256 = f1_score(y_true_256, y_pred_256, average='weighted')

# Calcular las métricas
accuracy_area = accuracy_score(y_true_area, y_pred_area)
precision_area = precision_score(y_true_area, y_pred_area, average='weighted')
recall_area = recall_score(y_true_area, y_pred_area, average='weighted')
f1_area = f1_score(y_true_area, y_pred_area, average='weighted')

# Calcular las métricas
accuracy_tincion = accuracy_score(y_true_tincion, y_pred_tincion)
precision_tincion = precision_score(y_true_tincion, y_pred_tincion, average='weighted')
recall_tincion = recall_score(y_true_tincion, y_pred_tincion, average='weighted')
f1_tincion = f1_score(y_true_tincion, y_pred_tincion, average='weighted')

# Calcular las métricas
accuracy_hist = accuracy_score(y_true_hist, y_pred_hist)
precision_hist = precision_score(y_true_hist, y_pred_hist, average='weighted')
recall_hist = recall_score(y_true_hist, y_pred_hist, average='weighted')
f1_hist = f1_score(y_true_hist, y_pred_hist, average='weighted')

# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------
print("\n\n")

# Imprimir las métricas
print("Accuracy 512:", accuracy_512)
print("Precision 512:", precision_512)
print("Recall 512:", recall_512)
print("F1-score 512:", f1_512)

print("\n")

# Imprimir las métricas
print("Accuracy 256:", accuracy_256)
print("Precision 256:", precision_256)
print("Recall 256:", recall_256)
print("F1-score 256:", f1_256)

print("\n")

# Imprimir las métricas
print("Accuracy area:", accuracy_area)
print("Precision area:", precision_area)
print("Recall area:", recall_area)
print("F1-score area:", f1_area)

print("\n")

# Imprimir las métricas
print("Accuracy tincion:", accuracy_tincion)
print("Precision tincion:", precision_tincion)
print("Recall tincion:", recall_tincion)
print("F1-score tincion:", f1_tincion)

print("\n")

# Imprimir las métricas
print("Accuracy hist:", accuracy_hist)
print("Precision hist:", precision_hist)
print("Recall hist:", recall_hist)
print("F1-score hist:", f1_hist)


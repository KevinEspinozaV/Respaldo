import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

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

# Definir las matrices como listas de listas
train1 = np.array([[1, 0, 0, 2], [0, 3, 0, 0], [0, 2, 1, 0], [0, 0, 0, 3]])
train2 = np.array([[2, 1, 0, 0], [1, 2, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0]])
train3 = np.array([[1, 2, 0, 0], [1, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]])
train4 = np.array([[3, 0, 0, 0], [1, 2, 0, 0], [0, 2, 1, 0], [0, 2, 1, 0]])
train5 = np.array([[2, 1, 0, 0], [0, 2, 1, 0], [0, 0, 3, 0], [0, 0, 0, 3]])
train6 = np.array([[1, 2, 0, 0], [1, 0, 2, 0], [0, 1, 2, 0], [0, 0, 0, 3]])
train7 = np.array([[2, 1, 0, 0], [0, 3, 0, 0], [0, 1, 2, 0], [0, 0, 1, 2]])
train8 = np.array([[2, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 2], [0, 0, 0, 3]])
train9 = np.array([[2, 1, 0, 0], [0, 2, 1, 0], [0, 0, 1, 2], [0, 0, 1, 2]])
train10 = np.array([[2, 1, 0, 0], [0, 3, 0, 0], [0, 2, 1, 0], [0, 0, 0, 3]])
train11 = np.array([[2, 1, 0, 0], [0, 2, 1, 0], [0, 1, 2, 0], [0, 0, 1, 2]])

# sumar matrices
resultado_suma = train1+ train2 + train3 + train4 + train5 + train6 + train7 + train8 + train9 + train10 + train11
print(resultado_suma)

# Obtener las etiquetas reales y las predicciones
y_true_sin, y_pred_sin = matriz_to_list(resultado_suma)

# Calcular las métricas
accuracy_sin = accuracy_score(y_true_sin, y_pred_sin)
precision_sin = precision_score(y_true_sin, y_pred_sin, average='weighted')
recall_sin = recall_score(y_true_sin, y_pred_sin, average='weighted')
f1_sin = f1_score(y_true_sin, y_pred_sin, average='weighted')

# Imprimir las métricas
print("Accuracy sin:", accuracy_sin)
print("Precision sin:", precision_sin)
print("Recall sin:", recall_sin)
print("F1-score sin:", f1_sin)


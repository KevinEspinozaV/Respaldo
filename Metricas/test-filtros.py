from scipy.stats import kruskal

# Ejemplo de tres grupos de datos que se desean comparar
accuracy_1 = [0.833, 0.583, 0.667, 0.667, 0.667]
accuracy_2 = [0.583, 0.750, 0.583, 0.500, 0.833]
accuracy_3 = [0.333, 0.667, 0.833, 0.667, 0.750]

# Realizar el test de Kruskal-Wallis
resultado = kruskal(accuracy_1, accuracy_2, accuracy_3)

# Obtener el valor p del test
valor_p = resultado.pvalue

# Imprimir el resultado
print("Valor p:", valor_p)
import numpy as np
from scipy.stats import wilcoxon, ttest_rel

# Valores de Accuracy para cada repetición del enfoque 1 y enfoque 2
accuracy_1 = [0.667, 0.583, 0.750, 0.667, 0.833]
accuracy_2 = [0.583, 0.417, 0.750, 0.583, 0.750]

# Valores de F1-score para cada repetición del enfoque 1 y enfoque 2
f1_score_1 = [0.625, 0.528, 0.743, 0.657, 0.831]
f1_score_2 = [0.576, 0.446, 0.743, 0.581, 0.737]

# Realizar la prueba de rangos con signo de Wilcoxon para comparar los dos enfoques
wilcoxon_accuracy, p_value_accuracy = wilcoxon(accuracy_1, accuracy_2)
wilcoxon_f1_score, p_value_f1_score = wilcoxon(f1_score_1, f1_score_2)

# Imprimir los resultados
print("Resultados de la comparación de Accuracy:")
print("Estadístico de la prueba de Wilcoxon:", wilcoxon_accuracy)
print("Valor p (p-value):", p_value_accuracy)

print("\nResultados de la comparación de F1-score:")
print("Estadístico de la prueba de Wilcoxon:", wilcoxon_f1_score)
print("Valor p (p-value):", p_value_f1_score)

# Realizar la prueba t de Student emparejada para comparar los dos enfoques
t_accuracy, p_value_accuracy = ttest_rel(accuracy_1, accuracy_2)
t_f1_score, p_value_f1_score = ttest_rel(f1_score_1, f1_score_2)

print("-"*100)

# Imprimir los resultados
print("\nResultados de la comparación de Accuracy:")
print("Estadístico t:", t_accuracy)
print("Valor p (p-value):", p_value_accuracy)

print("\nResultados de la comparación de F1-score:")
print("Estadístico t:", t_f1_score)
print("Valor p (p-value):", p_value_f1_score)

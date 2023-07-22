import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# DataFrame original
df = pd.DataFrame({
    "Grupo": ["Filtro Área", "Filtro Área", "Filtro Área", "Filtro Área", "Filtro Área",
              "Filtro Tinción", "Filtro Tinción", "Filtro Tinción", "Filtro Tinción", "Filtro Tinción",
              "Filtro Histograma", "Filtro Histograma", "Filtro Histograma", "Filtro Histograma", "Filtro Histograma"],
    "Accuracy": [0.833, 0.583, 0.667, 0.667, 0.667, 0.583, 0.750, 0.583, 0.500, 0.833, 0.417, 0.667, 0.833, 0.667, 0.750],
    "Recall": [0.833, 0.583, 0.667, 0.667, 0.667, 0.583, 0.750, 0.583, 0.500, 0.833, 0.417, 0.667, 0.833, 0.667, 0.750],
    "F1_Score": [0.837, 0.489, 0.642, 0.654, 0.692, 0.564, 0.757, 0.581, 0.433, 0.829, 0.342, 0.589, 0.813, 0.580, 0.731],
    "Precision": [0.750, 0.875, 0.833, 0.833, 0.750, 0.750, 0.929, 0.875, 0.833, 0.714, 0.450, 0.833, 0.875, 0.833, 0.714]
})

# Transformar el DataFrame a un formato "tidy"
df_tidy = df.melt(id_vars='Grupo', var_name='Métrica', value_name='Valor')

# Gráfico de barras apiladas utilizando Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x='Grupo', y='Valor', hue='Métrica', data=df_tidy, palette='muted', ci=None)
plt.xlabel('Grupo')
plt.ylabel('Valor')
plt.title('Gráfico de barras apiladas para métricas y grupos')
plt.legend(title='Métrica', title_fontsize='12', fontsize='10')
plt.tight_layout()
plt.show()

# Cargar las librerías
library(ggplot2)

# Crear el dataframe con los datos proporcionados
df <- data.frame(
  Grupo = rep(c("Filtro Área", "Filtro Tinción", "Filtro Histograma"), each = 5),
  Accuracy = c(0.833, 0.583, 0.667, 0.667, 0.667, 0.583, 0.750, 0.583, 0.500, 0.833, 0.417, 0.667, 0.833, 0.667, 0.750),
  Recall = c(0.833, 0.583, 0.667, 0.667, 0.667, 0.583, 0.750, 0.583, 0.500, 0.833, 0.417, 0.667, 0.833, 0.667, 0.750),
  F1_Score = c(0.837, 0.489, 0.642, 0.654, 0.692, 0.564, 0.757, 0.581, 0.433, 0.829, 0.342, 0.589, 0.813, 0.580, 0.731),
  Precision = c(0.750, 0.875, 0.833, 0.833, 0.750, 0.750, 0.929, 0.875, 0.833, 0.714, 0.450, 0.833, 0.875, 0.833, 0.714)
)

# Calcular los máximos por grupo y métrica
max_values <- df %>%
  group_by(Grupo) %>%
  summarise(across(Accuracy:Precision, max)) %>%
  pivot_longer(cols = -Grupo, names_to = "Metrica", values_to = "Valor")

# Cargar las librerías
library(tidyr)
library(ggplot2)

# Reorganizar el dataframe en un formato adecuado para ggplot2
df_long <- df %>% pivot_longer(cols = c(Accuracy, Recall, F1_Score, Precision),
                               names_to = "Metrica", values_to = "Valor")

# Crear el gráfico de barras con etiquetas
plot_metricas <- ggplot(df_long, aes(x = Grupo, y = Valor, fill = Metrica)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = "______"), position = position_dodge(width = 0.9), vjust = -0.1) +
  geom_text(data = max_values, aes(label = round(Valor, 3)), position = position_dodge(width = 0.9), vjust = -0.5) +
  labs(title = "Gráfico de barras - Métricas por Grupo",
       x = "Grupo",
       y = "Valor",
       fill = "Métrica") +
  theme_minimal()

# Mostrar el gráfico
print(plot_metricas)





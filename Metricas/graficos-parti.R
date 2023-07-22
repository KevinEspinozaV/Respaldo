# Cargar las librerías ggplot2 y dplyr
library(ggplot2)
library(dplyr)

# Crear el dataframe con los datos proporcionados
df <- data.frame(
  Grupo = rep(c("Particionamiento 512x512", "Particionamiento 256x256"), each = 5),
  Accuracy = c(0.667, 0.583, 0.750, 0.667, 0.833, 0.583, 0.417, 0.750, 0.583, 0.750),
  Recall = c(0.667, 0.583, 0.750,0.667, 0.833, 0.583, 0.417, 0.750, 0.583, 0.750),
  F1_Score = c(0.625, 0.528, 0.743, 0.657, 0.831, 0.576, 0.446, 0.743, 0.581, 0.737)
)

# Calcular los máximos por grupo y métrica
max_values <- df %>%
  group_by(Grupo) %>%
  summarise(across(Accuracy:F1_Score, max)) %>%
  pivot_longer(cols = -Grupo, names_to = "Metrica", values_to = "Valor")

# Convertir el dataframe de formato "wide" a "long" para que sea más fácil trazar el gráfico
df_long <- df %>% pivot_longer(cols = -Grupo, names_to = "Metrica", values_to = "Valor")

# Crear el gráfico de barras
ggplot(df_long, aes(x = Metrica, y = Valor, fill = Grupo)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = "__________"), position = position_dodge(width = 0.9), vjust = -0.1) +
  geom_text(data = max_values, aes(label = round(Valor, 3)), position = position_dodge(width = 0.9), vjust = -0.5) +
  theme_minimal() +
  labs(x = "Métrica", y = "Valor", title = "Gráfico de Barras por estrategia de particionamiento") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Cargar las librerías ggplot2 y dplyr
library(ggplot2)
library(dplyr)
library(tidyr)

# Crear el dataframe con los datos proporcionados
df <- data.frame(
  Grupo = rep(c("Agregación MAX", "Agregación Promedio"), each = 1),
  Accuracy = c(0.683, 0.667),
  Recall = c(0.683, 0.667),
  F1_Score = c(0.685, 0.670),
  Precision = c(0.727, 0.714)
)

# Calcular los máximos por grupo y métrica
max_values <- df %>%
  group_by(Grupo) %>%
  summarise(across(Accuracy:Precision, max)) %>%
  pivot_longer(cols = -Grupo, names_to = "Metrica", values_to = "Valor")

# Convertir el dataframe de formato "wide" a "long" para que sea más fácil trazar el gráfico
df_long <- df %>% pivot_longer(cols = -Grupo, names_to = "Metrica", values_to = "Valor")

# Crear el gráfico de barras
ggplot(df_long, aes(x = Metrica, y = Valor, fill = Grupo)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = "________"), position = position_dodge(width = 0.9), vjust = -0.1) +
  geom_text(data = max_values, aes(label = round(Valor, 3)), position = position_dodge(width = 0.9), vjust = -0.5) +
  theme_minimal() +
  labs(x = "Métrica", y = "Valor", title = "Gráfico de Barras por estrategia de particionamiento") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

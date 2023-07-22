library(ggplot2)
library(tidyr)

# Datos de ejemplo
tiempo_area <- c("11:03:14", "13:20:21", "13:46:05", "15:58:24", "14:26:28")
tiempo_hsv <- c("11:10:24", "10:28:25", "13:10:46", "16:30:35", "14:31:41")
tiempo_histo <- c("11:10:08", "10:36:01", "12:56:34", "15:55:10", "14:21:45")

# Convertir tiempos a formato de fecha y hora
tiempo_area <- as.POSIXct(tiempo_area, format = "%H:%M:%S")
tiempo_hsv <- as.POSIXct(tiempo_hsv, format = "%H:%M:%S")
tiempo_histo <- as.POSIXct(tiempo_histo, format = "%H:%M:%S")

# Crear un data.frame con los tiempos y las áreas
data <- data.frame(
  Tiempo = c(tiempo_area, tiempo_hsv, tiempo_histo),
  Area = rep(c("Filtro Área", "Filtro HSV", "Filtro Histo"), each = 5)
)

# Crear el gráfico de puntos con colores personalizados
ggplot(data, aes(x = Area, y = Tiempo, fill = Area)) +
  geom_point(shape = 21, color = "black", size = 3) +
  labs(x = "Estrategia de Filtrado", y = "Tiempo (HH:MM:SS)") +
  scale_fill_manual(values = c("Filtro Área" = "blue", "Filtro HSV" = "red", "Filtro Histo" = "green")) +  # Mostrar la leyenda
  scale_y_datetime(date_labels = "%H:%M:%S", date_breaks = "1 hour") +  # Formato HH:MM:SS y saltos de 1 hora
  theme_minimal()

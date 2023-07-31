library(ggplot2)
library(tidyr)

# Datos de ejemplo
tiempo_512 <- c("11:28:40",	"13:12:00", "13:08:00", "14:45:00",	"14:36:42")
tiempo_256 <-	c("19:12:39",	"20:35:54",	"19:03:09",	"20:52:38",	"19:15:27")

# Convertir tiempos a formato de fecha y hora
tiempo_512 <- as.POSIXct(tiempo_512, format = "%H:%M:%S")
tiempo_256 <- as.POSIXct(tiempo_256, format = "%H:%M:%S")

# Crear un data.frame con los tiempos y las áreas
data <- data.frame(
  Tiempo = c(tiempo_512, tiempo_256),
  Particionamiento = rep(c("Particionamiento 512x512", "Particionamiento 256x256" ), each = 5)
)

# Crear el gráfico de puntos con colores personalizados
ggplot(data, aes(x = Particionamiento, y = Tiempo, fill = Particionamiento)) +
  geom_point(shape = 21, color = "black", size = 3) +
  labs(x = "Estrategia de Particionamiento", y = "Cantidad de horas (HH:MM:SS)") +
  scale_fill_manual(values = c("Particionamiento 512x512" = "blue", "Particionamiento 256x256" = "red")) +  # Mostrar la leyenda
  scale_y_datetime(date_labels = "%H:%M:%S", date_breaks = "1 hour") +  # Formato HH:MM:SS y saltos de 1 hora
  theme_minimal()

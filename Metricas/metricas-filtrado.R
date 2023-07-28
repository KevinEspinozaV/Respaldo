# Cargar las librerías
library(ggplot2)

# Crear el dataframe con los datos proporcionados
datos_acc <- data.frame(
  grupo_acc = c(rep("FiltroÁrea", 5), rep("FiltroTinción", 5), rep("FiltroHistograma", 5)),
  valores_acc = c(0.833, 0.583, 0.667, 0.667, 0.667, 0.583, 0.750, 0.583, 0.500, 0.833, 0.417, 0.667, 0.833, 0.667, 0.750)
)

krskal_acc <- kruskal.test(valores_acc ~ grupo_acc, data = datos_acc)

# Crear el dataframe con los datos proporcionados
datos_pre <- data.frame(
  grupo_pre = c(rep("FiltroÁrea", 5), rep("FiltroTinción", 5), rep("FiltroHistograma", 5)),
  valores_pre = c(0.750, 0.875, 0.833, 0.833, 0.750, 0.750, 0.929, 0.875, 0.833, 0.714, 0.450, 0.833, 0.875, 0.833, 0.714)
)

krskal_pre <- kruskal.test(valores_pre ~ grupo_pre, data = datos_pre)

# Crear el dataframe con los datos proporcionados
datos_f1 <- data.frame(
  grupo_f1 = c(rep("FiltroÁrea", 5), rep("FiltroTinción", 5), rep("FiltroHistograma", 5)),
  valores_f1 = c(0.837, 0.489, 0.642, 0.654, 0.692, 0.564, 0.757, 0.581, 0.433, 0.829, 0.342, 0.589, 0.813, 0.580, 0.731)
)

krskal_f1 <- kruskal.test(valores_f1 ~ grupo_f1, data = datos_f1)

# Crear el dataframe con los datos proporcionados
datos_reca <- data.frame(
  grupo_reca = c(rep("FiltroÁrea", 5), rep("FiltroTinción", 5), rep("FiltroHistograma", 5)),
  valores_reca = c(0.833, 0.583, 0.667, 0.667, 0.667, 0.583, 0.750, 0.583, 0.500, 0.833, 0.417, 0.667, 0.833, 0.667, 0.750)
)

krskal_reca <- kruskal.test(valores_reca ~ grupo_reca, data = datos_reca)

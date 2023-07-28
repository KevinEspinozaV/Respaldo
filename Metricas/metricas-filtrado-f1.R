library(pwr)

# Crear el dataframe con los datos proporcionados
datos_f1 <- data.frame(
  grupo_f1 = c(rep("FiltroÁrea", 5), rep("FiltroTinción", 5), rep("FiltroHistograma", 5)),
  valores_f1 = c(0.837, 0.489, 0.642, 0.654, 0.692, 0.564, 0.757, 0.581, 0.433, 0.829, 0.342, 0.589, 0.813, 0.580, 0.731)
)

krskal_f1 <- kruskal.test(valores_f1 ~ grupo_f1, data = datos_f1)
print(krskal_f1)

# Parámetros para la simulación
n_simulaciones <- 1000  # Número de simulaciones a realizar
tamaño_muestra <- 5      # Tamaño de muestra en cada grupo
n_grupos <- 3            # Número de grupos

# Función para realizar el test de Kruskal-Wallis en datos simulados
realizar_simulacion_kruskal <- function(n_simulaciones, tamaño_muestra, n_grupos) {
  potencia <- replicate(n_simulaciones, {
    # Simular datos bajo el escenario nulo (misma distribución en cada grupo)
    simulacion_datos <- data.frame(
      grupo_f1 = rep(1:n_grupos, each = tamaño_muestra),
      valores_f1 = rnorm(tamaño_muestra * n_grupos)
    )
    
    # Realizar el test de Kruskal-Wallis en los datos simulados
    kruskal_result <- kruskal.test(valores_f1 ~ grupo_f1, data = simulacion_datos)
    
    # Almacenar si se rechazó o no la hipótesis nula (TRUE/FALSE)
    result_rechazo <- kruskal_result$p.value < 0.05
    
    return(result_rechazo)
  })
  
  # Calcular la potencia estadística
  potencia_estadistica <- mean(potencia)
  
  return(potencia_estadistica)
}

# Calcular la potencia estadística
potencia_result <- realizar_simulacion_kruskal(n_simulaciones, tamaño_muestra, n_grupos)
print(potencia_result)

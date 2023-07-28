library(pwr)

# Datos observados
datos_acc <- data.frame(
  grupo_acc = c(rep("FiltroÁrea", 5), rep("FiltroTinción", 5), rep("FiltroHistograma", 5)),
  valores_acc = c(0.833, 0.583, 0.667, 0.667, 0.667, 0.583, 0.750, 0.583, 0.500, 0.833, 0.417, 0.667, 0.833, 0.667, 0.750)
)

# Realizar el test de Kruskal-Wallis con los datos observados
krskal_acc <- kruskal.test(valores_acc ~ grupo_acc, data = datos_acc)
print(krskal_acc)

# Parámetros para la simulación
n_simulaciones <- 1000  # Número de simulaciones a realizar
tamaño_muestra <- 5      # Tamaño de muestra en cada grupo
n_grupos <- 3            # Número de grupos

# Función para realizar el test de Kruskal-Wallis en datos simulados
realizar_simulacion_kruskal <- function(n_simulaciones, tamaño_muestra, n_grupos) {
  potencia <- replicate(n_simulaciones, {
    # Simular datos bajo el escenario nulo (misma distribución en cada grupo)
    simulacion_datos <- data.frame(
      grupo_acc = rep(1:n_grupos, each = tamaño_muestra),
      valores_acc = rnorm(tamaño_muestra * n_grupos)
    )
    
    # Realizar el test de Kruskal-Wallis en los datos simulados
    kruskal_result <- kruskal.test(valores_acc ~ grupo_acc, data = simulacion_datos)
    
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

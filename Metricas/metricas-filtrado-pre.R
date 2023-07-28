library(pwr)

# Crear el dataframe con los datos proporcionados
datos_pre <- data.frame(
  grupo_pre = c(rep("FiltroÁrea", 5), rep("FiltroTinción", 5), rep("FiltroHistograma", 5)),
  valores_pre = c(0.750, 0.875, 0.833, 0.833, 0.750, 0.750, 0.929, 0.875, 0.833, 0.714, 0.450, 0.833, 0.875, 0.833, 0.714)
)

krskal_pre <- kruskal.test(valores_pre ~ grupo_pre, data = datos_pre)
print(krskal_pre)

# Parámetros para la simulación
n_simulaciones <- 1000  # Número de simulaciones a realizar
tamaño_muestra <- 5      # Tamaño de muestra en cada grupo
n_grupos <- 3            # Número de grupos

# Función para realizar el test de Kruskal-Wallis en datos simulados
realizar_simulacion_kruskal <- function(n_simulaciones, tamaño_muestra, n_grupos) {
  potencia <- replicate(n_simulaciones, {
    # Simular datos bajo el escenario nulo (misma distribución en cada grupo)
    simulacion_datos <- data.frame(
      grupo_pre = rep(1:n_grupos, each = tamaño_muestra),
      valores_pre = rnorm(tamaño_muestra * n_grupos)
    )
    
    # Realizar el test de Kruskal-Wallis en los datos simulados
    kruskal_result <- kruskal.test(valores_pre ~ grupo_pre, data = simulacion_datos)
    
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

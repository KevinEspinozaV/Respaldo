import cv2, os, glob
import numpy as np
import matplotlib.pyplot as plt

# Funcion encargada de guardar en una lista los parches que se leeran para filtrar
# entrada: ruta donde se encuentran los parches
# return: lista con el nombre de todos los parches 
def cargar_parches(patches_dir):

    # cargamos los parches que existen en la ruta
    if not os.path.exists(patches_dir):
        return []
    patches = glob.glob("{}/*.png".format(patches_dir), recursive=True)
    return patches # rutas en una lista

ruta = '/home/nyzcke/Escritorio/Aumento y Filtro/Filtrado/49_HER2'
ruta_parches = cargar_parches(ruta)

list_histogram = []
for parche in ruta_parches:

    # Cargar la imagen en formato BGR
    imagen_bgr = cv2.imread(parche)

    # Convertir la imagen a formato HSV
    imagen_hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)

    # Extraer el canal S
    canal_s = imagen_hsv[:, :, 1]

    # Calcular el histograma del canal S
    histograma = cv2.calcHist([canal_s], [0], None, [256], [0, 256])
    list_histogram.append(histograma)

# Calcular el histograma promedio
histograma_promedio = np.mean(list_histogram, axis=0)

# Mostrar el histograma promedio
plt.plot(histograma_promedio)
plt.title('Histograma promedio')
plt.xlabel('Valor del píxel')
plt.ylabel('Frecuencia')
plt.show()

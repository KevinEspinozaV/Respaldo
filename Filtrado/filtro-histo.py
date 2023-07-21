import cv2, os, glob, sys
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Modificar el índice del canal a 0 para el canal H
    hue_channel = hsv_image[:,:,0]

    # Calcular el histograma del canal H
    histogram = cv2.calcHist([hue_channel], [0], None, [180], [0, 180])

    return histogram.flatten()

def calculate_average_histogram(image_paths):
    histograms = []
    
    for image_path in image_paths:
        # Leer la imagen
        image = cv2.imread(image_path)
        
        # Calcular el histograma de la imagen
        histogram = calculate_histogram(image)
        
        # Agregar el histograma a la lista
        histograms.append(histogram)
    
    # Calcular el histograma promedio
    average_histogram = np.mean(histograms, axis=0)
    
    return average_histogram

# Funcion encargada de guardar en una lista los parches que se leeran para filtrar
# entrada: ruta donde se encuentran los parches
# return: lista con el nombre de todos los parches 
def cargar_parches(patches_dir):

    # cargamos los parches que existen en la ruta
    if not os.path.exists(patches_dir):
        return []
    patches = glob.glob("{}/*.png".format(patches_dir), recursive=True)
    return patches # rutas en una lista

# Funcion encargada de guardar en una lista las carpetas que se leeran para filtrar
# entrada: ruta donde se encuentran estas carpetas
# return: lista con el nombre de todas las carpetas leídas en orden de menor a mayor
def cargar_carpetas(ruta):

    carpetas = []
    # Por cada carpeta en la ruta
    for elemento in os.listdir(ruta):
        ruta_elemento = os.path.join(ruta, elemento) # guardo la ruta de la carpeta
        if os.path.isdir(ruta_elemento): # si existe la ruta
            carpetas.append(elemento) # guardo la ruta en la lista de "carpetas"

    carpetas = sorted(carpetas) # ordeno la lista de menor a mayor

    return carpetas

# Funcion encargada de capturar los histogramas generales por cada clase de WSI
# entrada: ruta donde se encuentran los WSI
# return: lista con los 4 histogramas por clase
def capturar_histrograma_wsi(ruta_carpetas_wsi):
    list_histo_wsi = []
    for carpeta_wsi in ruta_carpetas_wsi:
        ruta_carpeta = cargar_carpetas("/home/nyzcke/Escritorio/Dataset/Nivel Parche 512/"+carpeta_wsi)
        list_histo = []
        for carpeta in ruta_carpeta:
            # Lista de rutas de las imágenes
            image_paths = cargar_parches("/home/nyzcke/Escritorio/Dataset/Nivel Parche 512/"+carpeta_wsi+"/"+carpeta)
            # Calcular el histograma promedio
            average_histogram = calculate_average_histogram(image_paths)
            list_histo.append(average_histogram)

        average_histogram_total = np.mean(list_histo, axis=0)
        average_histogram_total_norm = cv2.normalize(average_histogram_total, average_histogram_total, 0, 1, cv2.NORM_MINMAX)
        list_histo_wsi.append(average_histogram_total_norm)

    return list_histo_wsi

# Funcion encargada de graficar los histogramas
# entrada: lista de los histogramas
# return: 
def graficar_histogramas(list_histo_wsi):

    cont = 0
    for histograma in list_histo_wsi:

        # Graficar el histograma promedio
        plt.plot(histograma, color='r')
        plt.title(f'Histograma promedio de WSI {cont}')
        plt.xlabel('Intensidad')
        plt.ylabel('Frecuencia')
        plt.show()

        cont = cont + 1

    return

# Funcion encargada de concatenar 2 listas en una sola lista
# entrada: name_wsi son todos los nombres posibles de las WSI en el dataset y calificacion_wsi son la calificacion de cada una
# return: una unica lista tanto con el nombre y la calificacion correspondiente
def concatenar_lista(name_wsi, calificacion_wsi):

    lista_calificaciones = []

    i = 0
    # Recorremos las listas y agregamos a la lista_calificaciones
    while i < len(name_wsi):
        lista_calificaciones.append([name_wsi[i],calificacion_wsi[i]])
        i = i + 1

    return lista_calificaciones

# Funcion encargada obtener la calificacion de las WSI analizadas en el train
# entrada: ruta_carpetas son el nombre en string de cada wsi y las listas que concatene con la funcion anterior
# return: una lista con la calificacion correspondiente a cada WSI analizada
def obtener_calificacion_wsi(ruta_carpetas, name_wsi, calificacion_wsi):

    lista_calificaciones = concatenar_lista(name_wsi, calificacion_wsi)

    lista_name = []

    for name in ruta_carpetas:
        lista_name.append(int(name[:2]))

    i = 0
    j = 0
    calificacion = []
    while i < len(lista_name):
        j = 0
        while j < len(lista_calificaciones):
            if lista_name[i] == lista_calificaciones[j][0]:
                calificacion.append(lista_calificaciones[j][1])
                j = len(lista_calificaciones)
            else:
                j = j + 1
        i = i + 1
    
    return calificacion


def filtrar_parches(ruta_parches, calificacion_real):

    list_filt = []
    cont = 0
    for parche in ruta_parches:
        # Leo la imagen
        image = cv2.imread(parche)
                
        # Calcular el histograma de la imagen
        histogram = calculate_histogram(image)

        # Se normalizan ambos histogramas
        histogram_norm = cv2.normalize(histogram, histogram, 0, 1, cv2.NORM_MINMAX)

        list_bhatt = []
        for hist in list_histo_wsi:

            # Calcular la correlación entre los histogramas
            bhatt = cv2.compareHist(hist, histogram_norm, cv2.HISTCMP_BHATTACHARYYA)
            list_bhatt.append(bhatt)

        min_bhatt = min(list_bhatt) # Saca el maximo de ihc
        posicion_ihc = list_bhatt.index(min_bhatt) # Determina en que posición. POS 0 = ihc 0, POS 1 = ihc 1, POS 2 = ihc 2, POS 3 = ihc 3

        # Reviso el ihc
        if calificacion_real == 0 or calificacion_real == 1:
            if posicion_ihc != 0 and posicion_ihc != 1:
                list_filt.append(parche)
                cont = cont + 1
        else:
            if posicion_ihc != 2 and posicion_ihc != 3:
                list_filt.append(parche)
                cont = cont + 1
                
    return list_filt, cont

def borrar_elementos(list_filt):

    for parche in list_filt:
        if os.path.exists(parche):
            nombre_archivo = os.path.splitext(parche)[0]
            # Borrar el archivo
            os.remove(nombre_archivo+'.png')
            os.remove(nombre_archivo+'.pkl')
        else:
            print("El archivo no existe")

    return
    
# ---------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- MAIN ------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# Calificacion IHC de cada carpeta (revisar orden de las carpetas)
ruta_carpetas = cargar_carpetas("/home/nyzcke/Escritorio/Memoria/Filtrado/Train_5")
name_wsi = [12, 14, 25, 26, 27, 29, 30, 32, 33, 34, 35, 36, 38, 39, 40, 46, 47, 49, 50, 52, 55, 57, 61, 63, 64, 66, 67, 68, 70, 73, 74, 79, 82, 83, 84, 86, 87, 88]
calificacion_wsi = [1, 1, 2, 2, 3, 0, 3, 1, 3, 1, 3, 2, 3, 0, 2, 0, 1, 2, 2, 0, 2, 0, 3, 2, 1, 0, 2, 0, 0, 0, 2, 1, 3, 3, 3, 1, 0, 1]
calificacion = obtener_calificacion_wsi(ruta_carpetas, name_wsi, calificacion_wsi)

# Capturo las carpetas WSI
print("Se leen las carpetas WSI y se calcula los histogramas clase...")
ruta_carpetas_wsi = cargar_carpetas("/home/nyzcke/Escritorio/Dataset/Nivel Parche 512/")
list_histo_wsi = capturar_histrograma_wsi(ruta_carpetas_wsi)
print("Histogramas por clase capturados")

# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------COMPARACION-------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
print("Comienza la comparación...")

# Inicio el ciclo de filtrado
i = 0
cont_final = 0
list_nombres_parches = []
for carpeta in ruta_carpetas:
    print(f"Analizando {carpeta} que tiene calificacion {calificacion[i]} ...")
    ruta_parches = cargar_parches("/home/nyzcke/Escritorio/Memoria/Filtrado/Train_1/"+str(carpeta))
    list_filt, cont = filtrar_parches(ruta_parches, calificacion[i])
    borrar_elementos(list_filt)
    print(f"Borre {cont} elementos")
    i = i + 1
    cont_final = cont_final + cont

print(f"Borre en total {cont_final} de parches")


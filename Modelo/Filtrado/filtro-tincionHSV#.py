import cv2, os, glob
import numpy as np

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

# Funcion encargada de guardar en una lista los parches que se leeran para filtrar
# entrada: ruta donde se encuentran los parches
# return: lista con el nombre de todos los parches 
def cargar_parches(patches_dir):

    # cargamos los parches que existen en la ruta
    if not os.path.exists(patches_dir):
        return []
    patches = glob.glob("{}/*.png".format(patches_dir), recursive=True)
    return patches # rutas en una lista

# Funcion encargada de calcular el porcentaje de tincion marron que tiene el parche
# entrada: ruta donde se encuentran los parches
# return: porcentaje de tincion marron que tiene el parche
def calculo_porcentaje_tincion(ruta_parche):

    # Leo la imagen
    imagen_parche = cv2.imread(ruta_parche)

    # Determino el rango de color donde se encontrará el color cafe/marron
    marron1 = np.array([0, 50, 20], np.uint8)
    marron2 = np.array([50, 255, 255], np.uint8)

    imagen_parche_HSV = cv2.cvtColor(imagen_parche, cv2.COLOR_BGR2HSV) # Se transforma de RGB a HSV
    maskMarron = cv2.inRange(imagen_parche_HSV, marron1, marron2) # Guardo aquellos pixeles en una mascara que presente los colores establecidos en el rango
    maskMarronvis = cv2.bitwise_and(imagen_parche, imagen_parche, mask= maskMarron) # Junto las mascara con la imagen original
    mask_gray = cv2.cvtColor(maskMarronvis, cv2.COLOR_BGR2GRAY) # Transformo la mascara a escala de grises

    # Calcular el porcentaje de píxeles teñidos de marrón
    total_pixeles = mask_gray.shape[0] * mask_gray.shape[1]
    pixeles_marrones = np.count_nonzero(mask_gray != 0)

    # Calculo el porcentaje respecto al total
    porcentaje_tincion_marron = (pixeles_marrones / total_pixeles) * 100
    
    return porcentaje_tincion_marron

# Funcion encargada de realizar el filtrado de los parches segun la calificacion que tiene WSI completo
# entrada: ruta donde se encuentran los parches y la calificacion original
# return: cuantos elementos borre
def filtrado_parches(ruta_parches, calificacion):

    cont = 0

    for patch in ruta_parches:
        if calificacion == 0: # IHC 0: x < 2%
            porcentaje = calculo_porcentaje_tincion(patch)
            if porcentaje > 1.5:
                # Verificar si el archivo existe antes de intentar borrarlo
                if os.path.exists(patch):
                    os.remove(patch)
                    os.remove(os.path.splitext(patch)[0]+'.pkl')
                    cont = cont + 1
                else:
                    print("El archivo no existe.")
        if calificacion == 1: # IHC 1: 2% < x < 10%
            porcentaje = calculo_porcentaje_tincion(patch)
            if not porcentaje > 1.5 and porcentaje < 10.0:
                # Verificar si el archivo existe antes de intentar borrarlo
                if os.path.exists(patch):
                    os.remove(patch)
                    os.remove(os.path.splitext(patch)[0]+'.pkl')
                    cont = cont + 1
                else:
                    print("El archivo no existe.")
        if calificacion == 2: # IHC 2: 10% < x < 20%
            porcentaje = calculo_porcentaje_tincion(patch)
            if not porcentaje > 10.0 and porcentaje < 30.0:
                # Verificar si el archivo existe antes de intentar borrarlo
                if os.path.exists(patch):
                    os.remove(patch)
                    os.remove(os.path.splitext(patch)[0]+'.pkl')
                    cont = cont + 1
                else:
                    print("El archivo no existe.")
        if calificacion == 3: # IHC 3: x > 30%
            porcentaje = calculo_porcentaje_tincion(patch)
            if porcentaje < 30.0:
                # Verificar si el archivo existe antes de intentar borrarlo
                if os.path.exists(patch):
                    os.remove(patch)
                    os.remove(os.path.splitext(patch)[0]+'.pkl')
                    cont = cont + 1
                else:
                    print("El archivo no existe.")

    return cont

def concatenar_lista(name_wsi, calificacion_wsi):

    lista_calificaciones = []

    i = 0
    while i < len(name_wsi):
        lista_calificaciones.append([name_wsi[i],calificacion_wsi[i]])
        i = i + 1

    return lista_calificaciones

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

# ---------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- MAIN ------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# Obtengo las carpetas
ruta_carpetas = cargar_carpetas("/home/nyzcke/Escritorio/Set Model/Modelo/HER2Classification/Train_1")

# Calificacion IHC de cada carpeta (revisar orden de las carpetas)
name_wsi = [12, 14, 25, 26, 27, 29, 30, 32, 33, 34, 35, 36, 38, 39, 40, 46, 47, 49, 50, 52, 55, 57, 61, 63, 64, 66, 67, 68, 70, 73, 74, 79, 82, 83, 84, 86, 87, 88]
calificacion_wsi = [1, 1, 2, 2, 3, 0, 3, 1, 3, 1, 3, 2, 3, 0, 2, 0, 1, 2, 2, 0, 2, 0, 3, 2, 1, 0, 2, 0, 0, 0, 2, 1, 3, 3, 3, 1, 0, 1]
calificacion = obtener_calificacion_wsi(ruta_carpetas, name_wsi, calificacion_wsi)

'''# Inicio el ciclo de filtrado
i = 0
for carpeta in ruta_carpetas:
    print(f"Analizando {carpeta} que tiene calificacion {calificacion[i]} ...")
    ruta_parches = cargar_parches(carpeta)
    cont = filtrado_parches(ruta_parches, calificacion[i])
    print(f"Borre {cont} elementos")
    i = i + 1
'''

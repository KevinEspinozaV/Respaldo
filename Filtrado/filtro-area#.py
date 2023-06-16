import cv2, os, glob
import numpy as np

# Funcion encargada de calcular los contornos de color azul y marron/cafe
# entrada: la imagen de un parche
# return: Devuele el area total que abarca los contornos azules y marrones
def calcular_area_colores(imagen):
    # Convertir la imagen a espacio de color HLS
    imagen_hls = cv2.cvtColor(imagen, cv2.COLOR_BGR2HLS)
    
    # Definir los rangos de color para el azul en el espacio HLS
    rango_bajo_azul = np.array([80, 30, 30], dtype=np.uint8)
    rango_alto_azul = np.array([150, 255, 255], dtype=np.uint8)
    
    # Definir los rangos de color para el marrón/café en el espacio HLS
    rango_bajo_marron = np.array([10, 20, 20], dtype=np.uint8)
    rango_alto_marron = np.array([50, 200, 200], dtype=np.uint8)
    
    # Aplicar una máscara para encontrar los píxeles azules
    mascara_azul = cv2.inRange(imagen_hls, rango_bajo_azul, rango_alto_azul)
    
    # Aplicar una máscara para encontrar los píxeles marrones/café
    mascara_marron = cv2.inRange(imagen_hls, rango_bajo_marron, rango_alto_marron)
    
    # Encontrar los contornos de las áreas detectadas de color azul
    contornos_azul, _ = cv2.findContours(mascara_azul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Encontrar los contornos de las áreas detectadas de color marrón/café
    contornos_marron, _ = cv2.findContours(mascara_marron, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar los contornos por área mínima
    area_minima = 10
    contornos_filtrados_azul = []
    for contour in contornos_azul:
        area = cv2.contourArea(contour)
        if area > area_minima:
            contornos_filtrados_azul.append(contour)

    # Filtrar los contornos por área mínima
    contornos_filtrados_marron = []
    for contour in contornos_marron:
        area = cv2.contourArea(contour)
        if area > area_minima:
            contornos_filtrados_marron.append(contour)
    
    # Calcular el área total de los contornos encontrados de color azul
    area_total_azul = 0
    for contorno in contornos_filtrados_azul:
        area = cv2.contourArea(contorno)
        area_total_azul += area
    
    # Calcular el área total de los contornos encontrados de color marrón/café
    area_total_marron = 0
    for contorno in contornos_filtrados_marron:
        area = cv2.contourArea(contorno)
        area_total_marron += area
    
    return area_total_azul, area_total_marron

# Funcion encargada de calcular la califiación del parche segun el porcentaje de area que tenga
# entrada: el porcentaje de azul y marron que presenta el parche respectivamente
# return: Devuele la calificación determinada
def calculo_calificacion(porcentaje_area_azul, porcentaje_area_marron):

    if porcentaje_area_azul > porcentaje_area_marron: # Posible 0, 1
        if porcentaje_area_azul >= 5 and porcentaje_area_marron <= 5: # Umbrales definidos segun la media obtenida
            calificacion = 0
        else: # En caso de que no presente suficiente membrana
            calificacion = 9
    else: # Posible 1, 2, 3
        if porcentaje_area_marron <= 25 and porcentaje_area_azul <= 10: # Umbrales definidos segun la media obtenida
            calificacion = 1
        elif porcentaje_area_marron >= 25 and porcentaje_area_marron <= 45 and porcentaje_area_azul <= 10: # Umbrales definidos segun la media obtenida
            calificacion = 2
        elif porcentaje_area_marron >= 40 and porcentaje_area_azul <= 5: # Umbrales definidos segun la media obtenida
            calificacion = 3
        else: # En caso de que no cumpla que ninguna calificacion descrita anteriormente
            calificacion = 9

    return calificacion

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

# Funcion encargada de obtener la lista de las calificaciones
# entrada: ruta donde se encuentran los parches
# return: lista con todas las calificaciones obtenidas 
def obtener_calificaciones(rutas_parches):
    list_calificacion = []
    # Por cada parche
    for patch in rutas_parches:
        # Cargo la imagen
        imagen_ejemplo = cv2.imread(patch)

        # Calcular el área total de los colores azules y marrones en la imagen
        area_total_azul, area_total_marron = calcular_area_colores(imagen_ejemplo)

        # Calcular el área total de la imagen original
        area_total_imagen = imagen_ejemplo.shape[0] * imagen_ejemplo.shape[1]

        # Calcular el porcentaje de área ocupada por los colores azules y marrones
        porcentaje_area_azul = (area_total_azul / area_total_imagen) * 100
        porcentaje_area_marron = (area_total_marron / area_total_imagen) * 100

        '''print("Porcentaje de área ocupada por los colores azules: {:.2f}%".format(porcentaje_area_azul))
        print("Porcentaje de área ocupada por los colores marrones: {:.2f}%".format(porcentaje_area_marron))
        print("Saturación de la imagen: ", saturacion_promedio)'''

        # Obtener la calificacion
        calificacion = calculo_calificacion(porcentaje_area_azul, porcentaje_area_marron)
        # Agregar a la lista de calificacion
        list_calificacion.append(calificacion)
    
    return list_calificacion

# Funcion encargada borrar aquellos parches que no pertenecen a la calificacion original
# entrada: ruta donde se encuentran los parches, lista de calificaciones y la calificacion real
# return: cuantos elementos borre
def filtrar_elementos(rutas_parches, list_calificacion, calificacion_real):

    i = 0
    cont = 0
    list_nombres_parches = []
    # Para los parches con calificacion 0
    if calificacion_real == 0:
        while i < len(rutas_parches):
            if list_calificacion[i] != 0 and list_calificacion[i] != 1: # Deja pasar aquellos parches que tienen calificacion 0 y 1
                # Verificar si el archivo existe
                if os.path.exists(rutas_parches[i]):
                    nombre_archivo = os.path.splitext(rutas_parches[i])[0]
                    # Borrar el archivo
                    os.remove(nombre_archivo+'.png')
                    os.remove(nombre_archivo+'.pkl')

                    # Sumo el cont
                    cont = cont + 1

                    # Agrego el parche borrado
                    list_nombres_parches.append(nombre_archivo)
                else:
                    print("El archivo no existe")
            i = i + 1
    # Para los parches con calificacion 1
    if calificacion_real == 1:
        while i < len(rutas_parches):
            if list_calificacion[i] != 1: # Deja pasar sólo aquellos parches con calificacion 1
                # Verificar si el archivo existe
                if os.path.exists(rutas_parches[i]):
                    nombre_archivo = os.path.splitext(rutas_parches[i])[0]
                    # Borrar el archivo
                    os.remove(nombre_archivo+'.png')
                    os.remove(nombre_archivo+'.pkl')
                    cont = cont + 1

                    # Agrego el parche borrado
                    list_nombres_parches.append(nombre_archivo)
                else:
                    print("El archivo no existe")
            i = i + 1
    # Para los parches con calificacion 2
    if calificacion_real == 2:
        while i < len(rutas_parches):
            if list_calificacion[i] != 2: # Deja pasar sólo aquellos parches con calificacion 2
                # Verificar si el archivo existe
                if os.path.exists(rutas_parches[i]):
                    nombre_archivo = os.path.splitext(rutas_parches[i])[0]
                    # Borrar el archivo
                    os.remove(nombre_archivo+'.png')
                    os.remove(nombre_archivo+'.pkl')
                    cont = cont + 1

                    # Agrego el parche borrado
                    list_nombres_parches.append(nombre_archivo)
                else:
                    print("El archivo no existe")
            i = i + 1
    # Para los parches con calificacion 3
    if calificacion_real == 3:
        while i < len(rutas_parches):
            if list_calificacion[i] != 3:  # Deja pasar sólo aquellos parches con calificacion 3
                # Verificar si el archivo existe
                if os.path.exists(rutas_parches[i]):
                    nombre_archivo = os.path.splitext(rutas_parches[i])[0]
                    # Borrar el archivo
                    os.remove(nombre_archivo+'.png')
                    os.remove(nombre_archivo+'.pkl')
                    cont = cont + 1

                    # Agrego el parche borrado
                    list_nombres_parches.append(nombre_archivo)
                else:
                    print("El archivo no existe")
            i = i + 1

    return cont, list_nombres_parches

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

def aplanar_lista(lista_anidada):

    lista_plana = []

    for sublista in lista_anidada:
        lista_plana.extend(sublista)

    return lista_plana

# CREAR FUNCION QUE BORRE LA FILA DEL CSV Y COMENTAR LAS WEAS 

# ---------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- MAIN ------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# Obtengo las carpetas
ruta_carpetas = cargar_carpetas("/home/nyzcke/Escritorio/Memoria/Filtrado/Train_1")
print(ruta_carpetas)
# Calificacion IHC de cada carpeta (revisar orden de las carpetas)
name_wsi = [12, 14, 25, 26, 27, 29, 30, 32, 33, 34, 35, 36, 38, 39, 40, 46, 47, 49, 50, 52, 55, 57, 61, 63, 64, 66, 67, 68, 70, 73, 74, 79, 82, 83, 84, 86, 87, 88]
calificacion_wsi = [1, 1, 2, 2, 3, 0, 3, 1, 3, 1, 3, 2, 3, 0, 2, 0, 1, 2, 2, 0, 2, 0, 3, 2, 1, 0, 2, 0, 0, 0, 2, 1, 3, 3, 3, 1, 0, 1]
calificacion = obtener_calificacion_wsi(ruta_carpetas, name_wsi, calificacion_wsi)

# Inicio el ciclo de filtrado
i = 0
ruta = "/home/nyzcke/Escritorio/Memoria/Filtrado/Train_1"
list_nombres_parches = []
for carpeta in ruta_carpetas:
    print(f"Analizando {carpeta} que tiene calificacion {calificacion[i]} ...")
    ruta_parches = cargar_parches("/home/nyzcke/Escritorio/Memoria/Filtrado/Train_1/"+str(carpeta))
    lista_calificaciones = obtener_calificaciones(ruta_parches)
    cont, list_nombres_parches_aux = filtrar_elementos(ruta_parches,lista_calificaciones, calificacion[i])
    print(f"Borre {cont} elementos")
    list_nombres_parches.append(list_nombres_parches_aux)
    i = i + 1

list_nombres_parches = aplanar_lista(list_nombres_parches)
print(len(list_nombres_parches))




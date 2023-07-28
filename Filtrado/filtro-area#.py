import cv2, os, glob, csv, sys, pickle, random
import numpy as np

random.seed(100)

# ----------------------------------------------------------------------------------------------------------------
# -------------------------------------------------FILTRO---------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

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
    rango_alto_marron = np.array([60, 200, 200], dtype=np.uint8)
    
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

    ''' # Para graficar
    imagen_contornos = imagen.copy()
    cv2.drawContours(imagen, contornos_filtrados_marron, -1, (0, 0, 255), 2)

    # Mostrar la imagen original y la imagen con los contornos
    cv2.imshow("Imagen original", imagen)
    cv2.imshow("Imagen con contornos", imagen_contornos)
    # Esperar a que se presione una tecla para cerrar las ventanas
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
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

    if porcentaje_area_marron <= 5: # Umbrales definidos segun la media obtenida
        calificacion = 0
    elif porcentaje_area_azul <= 20 and porcentaje_area_marron <= 10: # Umbrales definidos segun la media obtenida
        calificacion = 1
    elif porcentaje_area_azul <= 20 and porcentaje_area_marron >= 10 and porcentaje_area_marron <= 20: # Umbrales definidos segun la media obtenida
        calificacion = 2
    elif porcentaje_area_azul <= 10 and porcentaje_area_marron >= 20: # Umbrales definidos segun la media obtenida
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
            if list_calificacion[i] != 0 and list_calificacion[i] != 1: # Deja pasar aquellos parches no tienen calificacion 0 y 1
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
            if list_calificacion[i] != 0 and list_calificacion[i] != 1: # Deja pasar aquellos parches que no tienen calificacion 0 o 1
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
            if list_calificacion[i] != 2 and list_calificacion[i] != 3: # Deja pasar sólo aquellos parches que no tienen calificacion 2
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
            if list_calificacion[i] != 3 and list_calificacion[i] != 2:  # Deja pasar sólo aquellos parches que no tienen calificacion 3
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

def sacar_nombre_parche(list_nombres_parches):

    list_nombre_parches_final = []

    for nombre in list_nombres_parches:
        cadena = nombre
        partes = cadena.split('/')
        penultima_parte = partes[-2]
        ultima_parte = partes[-1]
        list_nombre_parches_final.append([penultima_parte, ultima_parte])

    return list_nombre_parches_final

def borrar_elementos(list_nombres_parches, ruta_csv):

    # Archivo de entrada y salida
    csv_file = ruta_csv
    ruta_sin_extension = os.path.splitext(ruta_csv)[0]
    filtered_csv_file = ruta_sin_extension+'_filtrado.csv'

    patches_to_keep = list_nombres_parches

    data = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)

        for row in reader:
            data.append(row)

    filtered_data = [row for row in data if [row[0], row[3]] not in patches_to_keep]

    with open(filtered_csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for row in filtered_data:
            writer.writerow(row)

    return

def calculo_media_porcentaje(lista_porcentajes):

    # Inicializar las sumas de porcentajes azules y marrones
    suma_azules = 0
    suma_marrones = 0

    # Iterar sobre cada par de porcentajes en la lista
    for par_porcentajes in lista_porcentajes:
        suma_azules += par_porcentajes[0]  # Sumar el porcentaje azul
        suma_marrones += par_porcentajes[1]  # Sumar el porcentaje marrón

    # Calcular el promedio de azules y marrones
    total_elementos = len(lista_porcentajes)
    promedio_azules = suma_azules / total_elementos
    promedio_marrones = suma_marrones / total_elementos

    print("Promedio de azules:", promedio_azules)
    print("Promedio de marrones:", promedio_marrones)

    return

def calculo_num_parches(ruta_carpetas):

    # Calculo de numero de parches
    i = 0
    num_parche = []
    for carpeta in ruta_carpetas:
        ruta_parches = cargar_parches("/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_0/"+str(carpeta))
        num_parche.append(len(ruta_parches))
        i = i + 1

    return num_parche

def calculo_diferencia(list_num_parches_ini,list_num_parches_fin):

    i = 0
    list_dif = []
    while i < len(list_num_parches_ini):
        dif = abs(list_num_parches_ini[i] - list_num_parches_fin[i])
        list_dif.append(dif)
        i = i + 1

    return list_dif

# ----------------------------------------------------------------------------------------------------------------
# -----------------------------------------AUMENTO DE DATOS-------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

def flip_images(image, direccion):
    
    if direccion == 1: # Horizontal
        # Voltea la imagen horizontalmente
        flipped_image = cv2.flip(image, 1)
    else: # Vertical
        flipped_image = cv2.flip(image, 0)

    return flipped_image

def flippingImagenes(parche,direccion,archivo_csv,calificacion):
    
    nuevos_datos = []
    
    imagen_original = cv2.imread(parche)

    if direccion == 0:
        img_flip = flip_images(imagen_original,0)
    else:
        img_flip = flip_images(imagen_original,1)

    # Guardar la imagen flippada
    name_patch = os.path.splitext(parche)[0]
    name_patch = name_patch.split("/")
    name_patch = name_patch[-1]
    name_patch = name_patch[:-3]
    ruta_png = '/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_0/'+carpeta+'/'+name_patch+'fp'+str(direccion)+'.png'
    ruta_pkl = '/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_0/'+carpeta+'/'+name_patch+'fp'+str(direccion)+'.pkl'
    try:
        if not os.path.exists(ruta_png):
            cv2.imwrite(ruta_png,img_flip)
            with open(ruta_pkl, 'wb') as f:
                pickle.dump(img_flip, f) # Creo el archivo .pkl
            if calificacion == 0 or calificacion ==1:
                nuevos_datos.append([carpeta, calificacion, 0, name_patch+'r'+str(direccion)])
            else:
                nuevos_datos.append([carpeta, calificacion, 1, name_patch+'r'+str(direccion)])
        else:
            return 0
    except:
        print("No pude guardar las imagenes")

    escribir_csv(archivo_csv, nuevos_datos)

    return 1

def rotate_image(img, angle):
    
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rotada = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return img_rotada

def rotacionImagenes(parche,carpeta,angulo_rotacion,archivo_csv, calificacion):

    nuevos_datos = []

    imagen_original = cv2.imread(parche)
    imagen_rotada = rotate_image(imagen_original, angulo_rotacion)

    # Guardar la imagen rotada y extendida
    name_patch = os.path.splitext(parche)[0]
    name_patch = name_patch.split("/")
    name_patch = name_patch[-1]
    name_patch = name_patch[:-3]
    ruta_png = '/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_0/'+carpeta+'/'+name_patch+'r'+str(angulo_rotacion)+'.png'
    ruta_pkl = '/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_0/'+carpeta+'/'+name_patch+'r'+str(angulo_rotacion)+'.pkl'
    try:
        if not os.path.exists(ruta_png):
            cv2.imwrite(ruta_png,imagen_rotada)
            with open(ruta_pkl, 'wb') as f:
                pickle.dump(imagen_rotada, f) # Creo el archivo .pkl
            if calificacion == 0 or calificacion ==1:
                nuevos_datos.append([carpeta, calificacion, 0, name_patch+'r'+str(angulo_rotacion)])
            else:
                nuevos_datos.append([carpeta, calificacion, 1, name_patch+'r'+str(angulo_rotacion)])
        else:
            return 0
    except:
        print("No pude guardar las imagenes")

    escribir_csv(archivo_csv, nuevos_datos)

    return 1

def escribir_csv(archivo_csv,nuevos_datos):

    # Leer el archivo CSV original
    with open(archivo_csv, 'r') as archivo_original:
        lector_csv = csv.reader(archivo_original)
        datos_originales = list(lector_csv)

    # Combinar los datos originales con los nuevos datos
    datos_combinados = datos_originales + nuevos_datos

    # Escribir los datos combinados en el archivo CSV original
    with open(archivo_csv, 'w', newline='') as archivo_actualizado:
        escritor_csv = csv.writer(archivo_actualizado)
        escritor_csv.writerows(datos_combinados)

    return

# ---------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- MAIN ------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# Obtengo las carpetas
ruta = "/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_0"
ruta_carpetas = cargar_carpetas(ruta)

# Calificacion IHC de cada carpeta (revisar orden de las carpetas)
name_wsi = [12, 14, 25, 26, 27, 29, 30, 32, 33, 34, 35, 36, 38, 39, 40, 46, 47, 49, 50, 52, 55, 57, 61, 63, 64, 66, 67, 68, 70, 73, 74, 79, 82, 83, 84, 86, 87, 88]
calificacion_wsi = [1, 1, 2, 2, 3, 0, 3, 1, 3, 1, 3, 2, 3, 0, 2, 0, 1, 2, 2, 0, 2, 0, 3, 2, 1, 0, 2, 0, 0, 0, 2, 1, 3, 3, 3, 1, 0, 1]
calificacion = obtener_calificacion_wsi(ruta_carpetas, name_wsi, calificacion_wsi)

# Antes de filtrar, total de parches:
list_num_parches_ini = calculo_num_parches(ruta_carpetas)

# Inicio el ciclo de filtrado
i = 0
list_nombres_parches = []
cont_parches_total = 0

cont_0 = 0
cont_1 = 0
cont_2 = 0
cont_3 = 0
for carpeta in ruta_carpetas:
    print(f"Analizando {carpeta} que tiene calificacion {calificacion[i]} ...")
    ruta_parches = cargar_parches("/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_0/"+str(carpeta))
    lista_calificaciones = obtener_calificaciones(ruta_parches)
    cont, list_nombres_parches_aux = filtrar_elementos(ruta_parches,lista_calificaciones, calificacion[i])
    list_nombres_parches.append(list_nombres_parches_aux)
    print(f"Borre {cont} parches de un total de {len(ruta_parches)}. Total: {len(ruta_parches)-cont}")
    cont_parches_total = cont_parches_total + len(ruta_parches)

    if calificacion[i] == 0:
        cont_0 = cont_0 + cont
    elif calificacion[i] == 1:
        cont_1 = cont_1 + cont
    elif calificacion[i] == 2:
        cont_2 = cont_2 + cont
    else:
        cont_3 = cont_3 + cont

    i = i + 1

list_parches_cont = [cont_0,cont_1,cont_2,cont_3]

print("\n\n")
print("Total Parches ",cont_parches_total)
print("Total Borrados ",sum(list_parches_cont))
print("Borrados por clase ", list_parches_cont)


#Despues de filtrar, total de parches
list_num_parches_fin = calculo_num_parches(ruta_carpetas)

# Obtengo las nuevos parches que quedan
list_nombres_parches = aplanar_lista(list_nombres_parches)
list_nombres = sacar_nombre_parche(list_nombres_parches)

print("\n\nEstoy borrando los elementos del csv...")
ruta_csv = "/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_0/train_0.csv"
borrar_elementos(list_nombres, ruta_csv)
print("Elementos borrados.\n\n")

list_dif = calculo_diferencia(list_num_parches_ini,list_num_parches_fin)

ruta_csv_nuevo = "/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_0/train_0_filtrado.csv"
i = 0
while i < len(list_dif): # Recorro la lista de dif
    if list_dif[i] != 0:
        print(f"Aumentando datos en la carpeta {ruta_carpetas[i]}")
        cont = 0
        while cont < list_dif[i]:
            ruta_parches = cargar_parches("/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_0/"+str(ruta_carpetas[i]))
            parche_aleatorio = random.choice(ruta_parches)
            lista_angulos = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345, 360]
            angulo_aleatorio = random.choice(lista_angulos)
            flag = rotacionImagenes(parche_aleatorio,ruta_carpetas[i],angulo_aleatorio,ruta_csv_nuevo, calificacion[i])
            
            if flag == 1:
                cont = cont + 1
        print(f"Aumente {cont} parches")
    i = i + 1

print("-"*120)

print("Antes de filtrar, total de parches: ")
print(list_num_parches_ini)

print("Despues de filtrar, total de parches: ")
print(list_num_parches_fin)

print("Despues de aumento de datos, total de parches: ")
list_num_parches_fin_2 = calculo_num_parches(ruta_carpetas)
print(list_num_parches_fin_2)
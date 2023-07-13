import cv2, glob, os, csv, random, pickle
import numpy as np
from scipy import spatial

# ---------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------CARGAR DATOS----------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

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

def cargar_img_train(train_dir):

    # Lista para almacenar las imágenes y etiquetas de entrenamiento
    train_images = []
    train_labels = []

    print("Cargando las imagenes de entrenamiento")
    # Cargar imágenes de entrenamiento
    for label in range(4):  # Considerando 4 clasificaciones (0, 1, 2, 3)
        label_dir = train_dir + '/' + str(label) + '/'
        images = glob.glob(label_dir + '/**/*.png')
        for image_path in images:
            # Cargar la imagen en formato BGR
            imagen_bgr = cv2.imread(image_path)

            # Convertir la imagen a formato HSV
            imagen_hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)

            # Extraer el canal S
            canal_s = imagen_hsv[:, :, 1]

            train_images.append(canal_s)
            train_labels.append(label)
    print("Imagenes de entrenamiento cargadas")

    return train_images, train_labels

def cargar_img_test(test_dir):
    test_images = []
    print("Cargando las imagenes de testing")
    for carpeta in ruta_carpetas:
        test_images_aux = calculo_histo_test(test_dir+'/'+carpeta)
        test_images.append(test_images_aux)
    print("Imagenes de testing cargadas")

    return test_images

def calculo_histo_test(test_dir):

    # Lista para almacenar las imágenes de prueba
    test_images = []
    # Cargar imágenes de prueba
    test_images_paths = glob.glob(test_dir + '/*.png')
    for image_path in test_images_paths:
        # Cargar la imagen en formato BGR
        imagen_bgr = cv2.imread(image_path)

        # Convertir la imagen a formato HSV
        imagen_hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)

        # Extraer el canal S
        canal_s = imagen_hsv[:, :, 1]

        test_images.append(canal_s)

    return test_images

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

def calculo_num_parches(ruta_carpetas):

    # Calculo de numero de parches
    i = 0
    ruta = "/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_5/"
    num_parche = []
    for carpeta in ruta_carpetas:
        ruta_parches = cargar_parches(ruta+str(carpeta))
        num_parche.append(len(ruta_parches))
        i = i + 1

    return num_parche

# ---------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------FILTRO----------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def calculo_list_classification(train_images,train_labels,test_images):

    # Calcular histograma de cada imagen de entrenamiento
    print("\nCalculando histograma de cada imagen de entrenamiento")
    train_hist = []
    for image in train_images:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])  # Calcular histograma
        hist = cv2.normalize(hist, hist).flatten()  # Normalizar y aplanar el histograma
        train_hist.append(hist)

    list_classification = []
    # Clasificar los parches de imágenes de prueba
    print("Clasificando los parches de imágenes de prueba")
    cont = 1
    for carpeta in test_images:
        print("Analizando la carpeta ",cont)
        list_classification_aux = []
        for test_image in carpeta:
            test_hist = cv2.calcHist([test_image], [0], None, [256], [0, 256])  # Calcular histograma del parche de prueba
            test_hist = cv2.normalize(test_hist, test_hist).flatten()  # Normalizar y aplanar el histograma del parche de prueba
            
            # Calcular similitud de histograma (correlación)
            similarities = [1 - spatial.distance.braycurtis(hist, test_hist) for hist in train_hist]
            
            # Obtener el índice de clasificación con la similitud máxima
            classification = train_labels[np.argmax(similarities)]
            #print("Parche de imagen clasificado como:", classification)
            list_classification_aux.append(classification)

        cont = cont + 1
        list_classification.append(list_classification_aux)
    
    return list_classification

def calculo_class_final(list_classification):

    list_class = []

    for lista in list_classification:
        cont_0 = 0
        cont_1 = 0
        cont_2 = 0
        cont_3 = 0
        for clas in lista:
            if clas == 0:
                cont_0 = cont_0 + 1
            elif clas == 1:
                cont_1 = cont_1 + 1
            elif clas == 2:
                cont_2 = cont_2 + 1
            else:
                cont_3 = cont_3 + 1

            list_cont = [cont_0,cont_1,cont_2,cont_3]
            
            max_cont = max(list_cont) # Saca el maximo
            posicion_cont = list_cont.index(max_cont) # Determina en que posición. POS 0 = ihc 0, POS 1 = ihc 1, POS 2 = ihc 2, POS 3 = ihc 3

            if posicion_cont == 0:
                classification = 0
            elif posicion_cont == 1:
                classification = 1
            elif posicion_cont == 2:
                classification = 2
            else: #posicion_cont == 3
                classification = 3
        
        print("Lista clasificaciones: ",list_cont)
        list_class.append(classification)

    return list_class


def filtrar_elementos(rutas_parches,list_calificacion,calificacion_real):

    i = 0
    cont = 0
    list_nombres_parches = []
    # Para los parches con calificacion 0
    if calificacion_real == 0:
        while i < len(rutas_parches):
            if list_calificacion[i] != 0: # Deja pasar aquellos parches no tienen calificacion 0 y 1 / and list_calificacion[i] != 1
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
            if list_calificacion[i] != 1: # Deja pasar aquellos parches que no tienen calificacion 0 o 1 / list_calificacion[i] != 0 and 
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
            if list_calificacion[i] != 2: # Deja pasar sólo aquellos parches que no tienen calificacion 2 / and list_calificacion[i] != 3
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
            if list_calificacion[i] != 3:  # Deja pasar sólo aquellos parches que no tienen calificacion 3 / list_calificacion[i] != 2 and 
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

def calculo_diferencia(list_num_parches_ini,list_num_parches_fin):

    i = 0
    list_dif = []
    while i < len(list_num_parches_ini):
        dif = abs(list_num_parches_ini[i] - list_num_parches_fin[i])
        list_dif.append(dif)
        i = i + 1

    return list_dif


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
    ruta_png = '/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_5/'+carpeta+'/'+name_patch+'r'+str(angulo_rotacion)+'.png'
    ruta_pkl = '/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_5/'+carpeta+'/'+name_patch+'r'+str(angulo_rotacion)+'.pkl'
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


# ---------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------MAIN---------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

# Directorios de imágenes de entrenamiento y prueba
train_dir = '/home/nyzcke/Escritorio/Test_5'
test_dir = '/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_5'

ruta_carpetas = cargar_carpetas(test_dir)

# Calificacion IHC de cada carpeta (revisar orden de las carpetas)
name_wsi = [12, 14, 25, 26, 27, 29, 30, 32, 33, 34, 35, 36, 38, 39, 40, 46, 47, 49, 50, 52, 55, 57, 61, 63, 64, 66, 67, 68, 70, 73, 74, 79, 82, 83, 84, 86, 87, 88]
calificacion_wsi = [1, 1, 2, 2, 3, 0, 3, 1, 3, 1, 3, 2, 3, 0, 2, 0, 1, 2, 2, 0, 2, 0, 3, 2, 1, 0, 2, 0, 0, 0, 2, 1, 3, 3, 3, 1, 0, 1]
calificacion = obtener_calificacion_wsi(ruta_carpetas, name_wsi, calificacion_wsi)

# Antes de filtrar, total de parches:
list_num_parches_ini = calculo_num_parches(ruta_carpetas)

# Cargamos imgagenes
train_images, train_labels = cargar_img_train(train_dir)
test_images = cargar_img_test(test_dir)

# Mostrar información sobre los datos cargados
print("Número de imágenes de entrenamiento:", len(train_images))
print("Número de etiquetas de entrenamiento:", len(train_labels))
print("Número de imágenes de prueba:", sum(len(sublista) for sublista in test_images))

# Calculo de lista clasificacion
list_classification = calculo_list_classification(train_images,train_labels,test_images)

# Calculo de los parches a borrar
i = 0
cont_parches_total = 0
list_nombres_parches = []
for carpeta in ruta_carpetas:
    print(f"Analizando {carpeta} que tiene calificacion {calificacion[i]} ...")
    ruta_parches = cargar_parches("/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_5/"+str(carpeta))
    cont, list_nombres_parches_aux = filtrar_elementos(ruta_parches,list_classification[i], calificacion[i])
    list_nombres_parches.append(list_nombres_parches_aux)
    print(f"Borre {cont} parches de un total de {len(ruta_parches)}. Total: {len(ruta_parches)-cont}")
    cont_parches_total = cont_parches_total + len(ruta_parches)
    i = i + 1

# --------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------BORRADO------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------

#Despues de filtrar, total de parches
list_num_parches_fin = calculo_num_parches(ruta_carpetas)

# Obtengo las nuevos parches que quedan
list_nombres_parches = aplanar_lista(list_nombres_parches)
list_nombres = sacar_nombre_parche(list_nombres_parches)

print("\n\nEstoy borrando los elementos del csv...")
ruta_csv = "/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_5/train_5.csv"
borrar_elementos(list_nombres, ruta_csv)
print("Elementos borrados.\n\n")

list_dif = calculo_diferencia(list_num_parches_ini,list_num_parches_fin)

ruta_csv_nuevo = "/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_5/train_5_filtrado.csv"
i = 0
while i < len(list_dif): # Recorro la lista de dif
    if list_dif[i] != 0:
        print(f"Aumentando datos en la carpeta {ruta_carpetas[i]}")
        cont = 0
        while cont < list_dif[i]:
            ruta_parches = cargar_parches("/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_5/"+str(ruta_carpetas[i]))
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
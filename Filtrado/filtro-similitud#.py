import cv2, glob, os
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
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Leer imagen en escala de grises
            train_images.append(image)
            train_labels.append(label)
    print("Imagenes de entrenamiento cargadas")

    return train_images, train_labels

def cargar_img_test(test_dir):

    # Lista para almacenar las imágenes de prueba
    test_images = []
    print("Cargando las imagenes de testing")
    # Cargar imágenes de prueba
    test_images_paths = glob.glob(test_dir + '/**/*.png')
    for image_path in test_images_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Leer imagen en escala de grises
        test_images.append(image)
    print("Imagenes de testing cargadas")

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
    train_hist = []
    for image in train_images:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])  # Calcular histograma
        hist = cv2.normalize(hist, hist).flatten()  # Normalizar y aplanar el histograma
        train_hist.append(hist)

    list_classification = []
    # Clasificar los parches de imágenes de prueba
    for test_image in test_images:
        test_hist = cv2.calcHist([test_image], [0], None, [256], [0, 256])  # Calcular histograma del parche de prueba
        test_hist = cv2.normalize(test_hist, test_hist).flatten()  # Normalizar y aplanar el histograma del parche de prueba
        
        # Calcular similitud de histograma (correlación)
        similarities = [1 - spatial.distance.braycurtis(hist, test_hist) for hist in train_hist]
        
        # Obtener el índice de clasificación con la similitud máxima
        classification = train_labels[np.argmax(similarities)]
        #print("Parche de imagen clasificado como:", classification)
        list_classification.append(classification)
    
    return list_classification

def calculo_class_final(list_classification):

    cont_0 = 0
    cont_1 = 0
    cont_2 = 0
    cont_3 = 0

    for clas in list_classification:
        if clas == 0:
            cont_0 = cont_0 + 1
        elif clas == 1:
            cont_1 = cont_1 + 1
        elif clas == 2:
            cont_2 = cont_2 + 1
        else:
            cont_3 = cont_3 + 1

    list_cont = [cont_0,cont_1,cont_2,cont_3]
    print("Lista clasificaciones: ",list_cont)

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

    return classification

# ---------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------MAIN---------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

# Directorios de imágenes de entrenamiento y prueba
train_dir = '/media/nyzcke/HDD/MEMORIA/Dataset/Train 512'
test_dir = '/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_5'

train_images, train_labels = cargar_img_train(train_dir)
test_images = cargar_img_test(test_dir)

# Mostrar información sobre los datos cargados
print("Número de imágenes de entrenamiento:", len(train_images))
print("Número de etiquetas de entrenamiento:", len(train_labels))
print("Número de imágenes de prueba:", len(test_images))

list_classification = calculo_list_classification(train_images,train_labels,test_images)
classification = calculo_class_final(list_classification)
print("La clasificación de la WSI es: ",classification)

# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
'''
# Obtengo las carpetas
ruta = "/home/nyzcke/Escritorio/Respaldo-main/Filtrado/Train_5"
ruta_carpetas = cargar_carpetas(ruta)

# Calificacion IHC de cada carpeta (revisar orden de las carpetas)
name_wsi = [12, 14, 25, 26, 27, 29, 30, 32, 33, 34, 35, 36, 38, 39, 40, 46, 47, 49, 50, 52, 55, 57, 61, 63, 64, 66, 67, 68, 70, 73, 74, 79, 82, 83, 84, 86, 87, 88]
calificacion_wsi = [1, 1, 2, 2, 3, 0, 3, 1, 3, 1, 3, 2, 3, 0, 2, 0, 1, 2, 2, 0, 2, 0, 3, 2, 1, 0, 2, 0, 0, 0, 2, 1, 3, 3, 3, 1, 0, 1]
calificacion = obtener_calificacion_wsi(ruta_carpetas, name_wsi, calificacion_wsi)

# Antes de filtrar, total de parches:
list_num_parches_ini = calculo_num_parches(ruta_carpetas,calificacion)

# Cargo las imagenes de train
train_images, train_labels = cargar_img_train(train_dir)

# Inicio el ciclo de filtrado
# REVISAR ESTOOO **** Como sacarlo por carpeta y no el total. Modificar la función cargar img test
test_images = cargar_img_test(test_dir)
'''
import cv2, os, glob, sys
import numpy as np

def detectar_colores_azules(imagen):
    # Convertir la imagen a espacio de color HLS
    imagen_hls = cv2.cvtColor(imagen, cv2.COLOR_BGR2HLS)
    
    # Definir los rangos de color para el azul en el espacio HLS
    rango_bajo = np.array([80, 30, 30], dtype=np.uint8)
    rango_alto = np.array([150, 240, 240], dtype=np.uint8)
    
    # Aplicar una máscara para encontrar los píxeles azules
    mascara = cv2.inRange(imagen_hls, rango_bajo, rango_alto)
    
    # Aplicar la máscara a la imagen original
    resultado = cv2.bitwise_and(imagen, imagen, mask=mascara)
    
    return resultado

def detectar_colores_cafes(imagen):
    # Convertir la imagen a espacio de color HLS
    imagen_hls = cv2.cvtColor(imagen, cv2.COLOR_BGR2HLS)
    
    # Definir los rangos de color para el café/marrón en el espacio HLS
    rango_bajo = np.array([10, 20, 20], dtype=np.uint8)
    rango_alto = np.array([50, 200, 200], dtype=np.uint8)
    
    # Aplicar una máscara para encontrar los píxeles cafés/marrones
    mascara = cv2.inRange(imagen_hls, rango_bajo, rango_alto)
    
    # Aplicar la máscara a la imagen original
    resultado = cv2.bitwise_and(imagen, imagen, mask=mascara)
    
    return resultado

def contar_pixeles(imagen):
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbralización para obtener una imagen binaria
    _, imagen_binaria = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY)

    # Contar los píxeles cafés en la imagen binaria
    cantidad_pixeles = cv2.countNonZero(imagen_binaria)
    
    return cantidad_pixeles

def cargar_parches(patches_dir):

    # load all WSI from patches_dir
    if not os.path.exists(patches_dir):
        return []
    patches = glob.glob("{}/*.png".format(patches_dir), recursive=True)
    return patches # rutas en una lista

def calcular_media(lista):
    suma = sum(lista)
    media = suma / len(lista)
    return media

# ----------------------------------------------------------------------------------------------
# ---------------------------------------- MAIN ------------------------------------------------
# ----------------------------------------------------------------------------------------------

patches_dir = "/home/nyzcke/Escritorio/Memoria/Filtrado/1"
rutas_parches = cargar_parches(patches_dir)

list_calificacion = []
for patch in rutas_parches:
    imagen_ejemplo = cv2.imread(patch)

    # Detectar colores azules en la imagen
    resultado = detectar_colores_azules(imagen_ejemplo)
    resultado2 = detectar_colores_cafes(imagen_ejemplo)

    # Contar pixeles
    pixeles_azules = contar_pixeles(resultado)
    pixeles_cafes = contar_pixeles(resultado2)
    altura, anchura, _ = resultado.shape
    pixeles_totales = altura * anchura

    # Porcentajes
    porc_pixeles_azules = (pixeles_azules/pixeles_totales)*100
    porc_pixeles_cafes = (pixeles_cafes/pixeles_totales)*100

    print(f"porc azul: {porc_pixeles_azules} y porc marron: {porc_pixeles_cafes}")

    if porc_pixeles_azules >= 8 and porc_pixeles_cafes <= 5:
        calificacion = 0
        list_calificacion.append(calificacion)
    elif porc_pixeles_azules >= 5 and (porc_pixeles_cafes >= 5 and porc_pixeles_cafes <= 8):
        calificacion = 1
        list_calificacion.append(calificacion)
    elif porc_pixeles_azules >= 5 and (porc_pixeles_cafes >= 8 and porc_pixeles_cafes <= 20):
        calificacion = 2
        list_calificacion.append(calificacion)
    elif porc_pixeles_azules >= 5 and porc_pixeles_cafes >= 20:
        calificacion = 3
        list_calificacion.append(calificacion)
    else:
        calificacion = "x"
        list_calificacion.append(calificacion)

print(list_calificacion)


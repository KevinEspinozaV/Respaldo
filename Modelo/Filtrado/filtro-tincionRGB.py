import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread("3/1_6144_22528_512.png")

cv2.imshow("original", image)

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar umbral adaptativo
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Encontrar los contornos
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crear una máscara en blanco del tamaño de la imagen
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Dibujar los contornos en la máscara
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

# Filtrar los contornos por área mínima
area_minima = 10
contornos_filtrados = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > area_minima:
        contornos_filtrados.append(contour)

# Inicializar una lista para almacenar las medias de color de los contornos
contour_colors = []

# Calcular la media del color para cada contorno encontrado
for contorno in contornos_filtrados:
    # Extraer el contorno como una región de interés (ROI) de la imagen original
    x, y, w, h = cv2.boundingRect(contorno)
    roi = image[y:y+h, x:x+w]

    # Calcular la media del color en cada canal de color (BGR)
    b_mean = np.mean(roi[:, :, 0])
    g_mean = np.mean(roi[:, :, 1])
    r_mean = np.mean(roi[:, :, 2])
    
    # Guardar la media del color en una lista
    contour_colors.append([b_mean, g_mean, r_mean])

'''# Imprimir las medias de color de los contornos
for i, color in enumerate(contour_colors):
    print(f"Contorno {i+1}: Media del color (B, G, R) = {color}")'''

cv2.drawContours(image,contornos_filtrados,-1,(0,0,255), 2)
cv2.imshow("contornos", image)
cv2.waitKey(0)

color_azul = 0
color_marron = 0
color_x = 0
for color in contour_colors:
    if (color[0] >= 0 and color[0] <= 90) and (color[1] >= 60 and color[1] <= 140) and (color[2] >= 100 and color[2] <= 190): # Color "cafe"
        color_marron = color_marron + 1
    elif (color[0] >= 100 and color[0] <= 255) and (color[1] >= 0 and color[1] <= 190) and (color[2] >= 0 and color[2] <= 190): # Color "azul"
        color_azul = color_azul + 1
    else:
        color_x = color_x + 1

print(f"color azul: {color_azul}, color marron: {color_marron} y color x: {color_x}")
print(f"Contornos: {len(contornos_filtrados)}")

porcentaje_azul = (color_azul*100)/len(contornos_filtrados)
porcentaje_marron = (color_marron*100)/len(contornos_filtrados)
porcentaje_x = (color_x*100)/len(contornos_filtrados)

print(f"Porcentaje azul: {porcentaje_azul}, porcentaje marron: {porcentaje_marron} y porcentaje x: {porcentaje_x}")

calficacion = 0
if len(contornos_filtrados) >= 80: # Puede ser 0, 1, 2
    if porcentaje_azul >= 30 and (porcentaje_marron >= 5 or porcentaje_x >= 5): # puede ser 2
        calficacion = 2
    elif porcentaje_azul >= 50 and (porcentaje_marron <= 5 or porcentaje_x <= 5): # puede ser 1
        calficacion = 1
    elif porcentaje_azul >= 60:
        calficacion = 0    
else:
    if porcentaje_marron >= 30:
        calficacion = 3         

print("Calificacion: ", calficacion)




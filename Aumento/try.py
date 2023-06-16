from math import floor
from PIL import Image
import random, csv, os, shutil, glob, math, cv2

random.seed(65)

def rotar_imagen(imagen, angulo):
    return imagen.rotate(angulo, resample=Image.BILINEAR, expand=True)

def flip_images(image, direccion):
    
    if direccion == 1: # Horizontal
        # Voltea la imagen horizontalmente
        flipped_image = cv2.flip(image, 1)
    else: # Vertical
        flipped_image = cv2.flip(image, 0)

    return flipped_image

def rotate_image(img, angle):
    
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rotada = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return img_rotada

def scale_image(image, scale_factor):

    if scale_factor < 1:
        height, width = image.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Resizing the image using bilinear interpolation
        scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Cropping or padding the scaled image to the original size
        top = max((height - new_height) // 2, 0)
        bottom = max(height - new_height - top, 0)
        left = max((width - new_width) // 2, 0)
        right = max(width - new_width - left, 0)
        scaled_image = cv2.copyMakeBorder(scaled_image, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE, value=0)    
        return scaled_image
    else:
        desired_size = (512,512)
        # Obtener las dimensiones originales de la imagen
        original_width, original_height = image.size
        
        # Calcular las nuevas dimensiones después de escalar
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Calcular el relleno necesario para alcanzar el tamaño deseado
        padding_width = max(0, desired_size[0] - new_width) // 2
        padding_height = max(0, desired_size[1] - new_height) // 2
        
        # Redimensionar la imagen con relleno
        scaled_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Crear una nueva imagen con el tamaño deseado y rellenarla con el fondo
        new_image = Image.new("RGB", desired_size, (255, 255, 255))
        new_image.paste(scaled_image, (padding_width, padding_height))

        return new_image

def rotacionImagenes(rutas,angulo_rotacion):

    for parche in rutas:
        imagen_original = cv2.imread(parche)
        imagen_rotada = rotate_image(imagen_original, angulo_rotacion)

        # Guardar la imagen rotada y extendida
        name_patch = os.path.splitext(parche)[0]
        name_patch = name_patch[8:]
        cv2.imwrite('Patches_Aumentado/'+name_patch+'_r'+str(angulo_rotacion)+'.png',imagen_rotada)

    return

def flippingImagenes(rutas,direccion):
    
    for parche in rutas:
        imagen_original = cv2.imread(parche)

        if direccion == "vertical":
            img_flip = flip_images(imagen_original,0)
        else:
            img_flip = flip_images(imagen_original,1)

        # Guardar la imagen rotada y extendida
        name_patch = os.path.splitext(parche)[0]
        name_patch = name_patch[8:]
        cv2.imwrite('Patches_Aumentado/'+name_patch+'_f'+str(direccion)+'.png',img_flip)

    return

def scallingImagenes(rutas,scale_factor):

    for parche in rutas:

        if scale_factor < 1:
            imagen_original = cv2.imread(parche)
            imagen_escalada = scale_image(imagen_original, scale_factor)

            # Guardar la imagen extendida
            name_patch = os.path.splitext(parche)[0]
            name_patch = name_patch[8:]
            cv2.imwrite(name_patch+'_sc_'+str(scale_factor)+'.png',imagen_escalada)
        else:
            imagen_original = Image.open(parche)
            imagen_escalada = scale_image(imagen_original, scale_factor)

            # Guardar la imagen extendida
            name_patch = os.path.splitext(parche)[0]
            name_patch = name_patch[8:]
            imagen_escalada.save('Patches_Aumentado/'+name_patch+'_sc_'+str(scale_factor)+'.png')

  
def load_patches(patches_dir):
    # load all WSI from patches_dir
    if not os.path.exists(patches_dir):
        return []
    patches = glob.glob("{}/**/*.png".format(patches_dir), recursive=True)
    return patches

# Main 

rutas_parches = load_patches("Patches/") 

carpeta = 'Patches_Aumentado'

# Verificar si la carpeta no existe antes de crearla
if not os.path.exists(carpeta):
    os.mkdir(carpeta)


rotacionImagenes(rutas_parches,angulo_rotacion=30)
flippingImagenes(rutas_parches,direccion="vertical")
flippingImagenes(rutas_parches,direccion="horizontal")
scallingImagenes(rutas_parches,scale_factor=1.25)

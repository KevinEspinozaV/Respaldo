import json,sys
import matplotlib.pyplot as plt

archivo = '/home/nyzcke/Escritorio/Respaldo dataset y check/checkpoint/Iter 2/her2classification/20230516_172638.log.json'  # Reemplaza 'ruta_del_archivo.json' con la ruta de tu archivo

with open(archivo, 'r') as f:
    lista_loss_epoch = []
    lista_prom_epoch = []
    lista_loss_val = []

    contador = 0
    for linea in f:
        if contador > 0:
            json_objeto = json.loads(linea) # Tomo la linea
            if json_objeto["mode"] == "train": # Pregunto si esta en train

                iter_aux = json_objeto["iter"] # Capturo el valor de la iteracion
                if iter_aux <= 345: # si no sobre pasa el max de iteraciones
                    lista_loss_epoch.append(json_objeto["loss_ihc"]) # guardo el valor en una lista

            if json_objeto["mode"] == "val": # Pregunto si esta en train

                suma = sum(lista_loss_epoch) # Sumo los valores que estan en la lista
                promedio = suma / len(lista_loss_epoch) # divido por el total para sacar el promedio
                lista_prom_epoch.append(promedio) # guardo el resultado promedio por iteraciones

                lista_loss_val.append(json_objeto["loss_ihc"]) # guardo el valor del loss obtenido en la validacion

                lista_loss_epoch = [] # Reinicio la lista de los loss por iteracion
                
        contador = contador + 1

# Datos de ejemplo
x = list(range(1, 101)) # Epoch
y1 = lista_prom_epoch # loss promedio por epoch
y2 = lista_loss_val # loss por val

# Crear el gráfico
plt.plot(x, y1, label='loss train')
plt.plot(x, y2, label='loss val')

# Personalizar el gráfico
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and validation loss')
plt.legend()

# Mostrar el gráfico
plt.show()

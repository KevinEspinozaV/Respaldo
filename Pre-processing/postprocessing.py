from math import floor
from PIL import Image
import random, csv, os, shutil

random.seed(100)

def definirListaIteracione(list_wsi):

    list_iter = []
    for i in range(5):
        list_iter_aux = []
        for lista in list_wsi:
            listaAux = []

            largo = len(lista) # Calculo el largo de la lista
            porcion_1 = floor(largo*0.7)

            valores_aleatorios = random.sample(lista, porcion_1)
            valores_no_seleccionados = list(set(lista) - set(valores_aleatorios))

            valores_aleatorios.sort()
            valores_no_seleccionados.sort()

            listaAux.append(valores_aleatorios)
            listaAux.append(valores_no_seleccionados)

            list_iter_aux.append(listaAux)
        list_iter.append(list_iter_aux)
    
    return list_iter

def calculoData(list_iter):
    list_data_aux = []
    for iter in list_iter:
        lista_aux_train = []
        lista_aux_test = []
        for iteraux in iter:
            lista_aux_train.append(iteraux[0])
            lista_aux_test.append(iteraux[1])
        list_data_aux.append([lista_aux_train, lista_aux_test])

    i = 0
    lista_data = []
    while i < len(list_data_aux): # Se recorre todas las iter
        lista_train_aux = []
        for lista in list_data_aux[i][0]:
            lista_train_aux.extend(lista)
        lista_test_aux = []
        for lista in list_data_aux[i][1]:
            lista_test_aux.extend(lista)
        lista_data.append([lista_train_aux,lista_test_aux])
        i = i + 1

    return lista_data

# Escribimos los archivos train y test csv
def crearArchivosCsv(list_data,key):
    # abrir el archivo de entrada CSV
    with open('data_info.csv', 'r') as input_file:
        # leer los datos del archivo de entrada CSV
        csv_reader = csv.reader(input_file)
        data = [row for row in csv_reader]

    encabezado = ["slide_bn","ihc","fish","patch_name"]

    # abrir el archivo de salida CSV
    cont = 1
    for list in list_data:
        if key == "train":
            nombre_archivo = 'train_'+str(cont)+'.csv'
        else:
            nombre_archivo = 'test_'+str(cont)+'.csv'
        with open(nombre_archivo, 'w', newline='') as output_file:
            # crear un objeto de escritura CSV
            csv_writer = csv.writer(output_file)
            csv_writer.writerow(encabezado)

            # escribir los datos en el archivo de salida CSV
            if key == "train":
                i = 0
                while i < len(list[0]):
                    for row in data:
                        nombre = row[0][:2]
                        if nombre == list[0][i]:
                            csv_writer.writerow(row)
                    i = i + 1
            if key == "test":
                i = 0
                while i < len(list[1]):
                    for row in data:
                        nombre = row[0][:2]
                        if nombre == list[1][i]:
                            csv_writer.writerow(row)
                    i = i + 1
        cont = cont + 1

    return

def moverCsv(list_data):
    # ruta del archivo a mover
    ruta_csv_train = []
    ruta_csv_test = []
    cont = 1
    for lista in list_data:
        ruta_csv_train.append('/home/nyzcke/Escritorio/Set Model/Pre-processing/train_'+str(cont)+'.csv') 
        ruta_csv_test.append('/home/nyzcke/Escritorio/Set Model/Pre-processing/test_'+str(cont)+'.csv')
        cont = cont + 1

    ruta_csv_data = '/home/nyzcke/Escritorio/Set Model/Pre-processing/data_info.csv'

    # ruta de la carpeta destino
    ruta_train = '/home/nyzcke/Escritorio/Set Model/Modelo/HER2Classification/Train'
    ruta_test = '/home/nyzcke/Escritorio/Set Model/Modelo/HER2Classification/Test'
    ruta_data = '/home/nyzcke/Escritorio/Set Model/Modelo/HER2Classification'

    # muevo los archivos csv de train 
    cont = 1
    for ruta in ruta_csv_train:
        if os.path.isfile(ruta): # Revsi
            shutil.move(ruta, ruta_train+'_'+str(cont)+'/train_'+str(cont)+'.csv') # Muevo el csv train
        cont = cont + 1

    cont = 1
    # muevo los archivos csv de test 
    for ruta in ruta_csv_test:      
        if os.path.isfile(ruta):
            shutil.move(ruta, ruta_test+'_'+str(cont)+'/test_'+str(cont)+'.csv')  # Muevo el csv test
        cont = cont + 1

    shutil.copy2(ruta_csv_data, ruta_data) # Muevo una copia del csv data

    return 

def moverParches(list_data):
    ruta_train = '/home/nyzcke/Escritorio/Set Model/Modelo/HER2Classification/Train'
    ruta_test = '/home/nyzcke/Escritorio/Set Model/Modelo/HER2Classification/Test'

    # Mueve las carpetas del WSI al train
    cont = 1
    for lista in list_data:
        for i in range(len(lista[0])):
            ruta_origen = '/home/nyzcke/Escritorio/Set Model/Pre-processing/.slide_patch_cache/'+lista[0][i]+'_HER2'
            try:
                rutaAux = ruta_train+'_'+str(cont)+'/'+lista[0][i]+'_HER2'
                shutil.copytree(ruta_origen, rutaAux) 
            except:
                print(f"la carpeta {ruta_origen} no está")
        cont = cont + 1
    
    # Mueve las carpetas al test
    cont = 1
    for lista in list_data:
        for i in range(len(lista[1])):
            ruta_origen = '/home/nyzcke/Escritorio/Set Model/Pre-processing/.slide_patch_cache/'+lista[1][i]+'_HER2'
            try:
                rutaAux = ruta_test+'_'+str(cont)+'/'+lista[1][i]+'_HER2'
                shutil.copytree(ruta_origen, rutaAux)
            except:
                print(f"la carpeta {ruta_origen} no está")
        cont = cont + 1
    return

#  --------------------------------------------------- MAIN ---------------------------------------------------

wsi_0 = ['29','39','46','52','57','66','68','70','73','87'] 
wsi_1 = ['12','14','32','34','47','64','79','86','88']
wsi_2 = ['25','26','36','40','49','50','55','63','67','74']
wsi_3 = ['27','30','33','35','38','61','82','83','84']

list_wsi = [wsi_0,wsi_1,wsi_2,wsi_3]

# Creamos los conjuntos randoms (Train 70% - Test 30%)
print("Establecemos los conjuntos de train y test")
list_iter = definirListaIteracione(list_wsi)
list_data = calculoData(list_iter)
print("Conjuntos creados correctamente\n")

# Creamos los archivos
try:
    print("Creamos los archivos CSV por cada train y test")
    crearArchivosCsv(list_data,"train")
    crearArchivosCsv(list_data,"test")
except:
    print("Existió algun error al generar los archivos CSV\n")

# Movemos los archivos
try:
    print("Movemos los parches")
    moverParches(list_data)
    print("Movemos los csv")
    moverCsv(list_data)
except:
    print("Existió en un problema al mover los parches")

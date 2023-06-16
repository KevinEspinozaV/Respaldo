import pickle

archivo_pkl = 'results.pkl'  # Reemplaza 'ruta_del_archivo.pkl' con la ruta de tu archivo .pkl
archivo_txt = 'results.txt'  # Reemplaza 'ruta_del_archivo.txt' con la ruta deseada para el archivo .txt

# Cargar el archivo .pkl
with open(archivo_pkl, 'rb') as f:
    datos = pickle.load(f)

# Guardar los datos en un archivo de texto
with open(archivo_txt, 'w') as f:
    for dato in datos:
        f.write(str(dato) + '\n')

import argparse
import os, sys, csv
import warnings

from itertools import accumulate

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

from sklearn.metrics import confusion_matrix, accuracy_score

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config) # Carga los config del archivo config
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None): # En caso de presentar custom imports
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline) 
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    
    '''with open('resultados.txt', 'w') as f:
        for data in data_loader:
            f.write(str(data))
    f.close()'''

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        print("entre a single gpu")
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        print("entre a multi gpu")
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    # outputs: son las salidas por cada parche

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
            print("\nResults written successfully")
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))

    return data_loader, outputs


# Cuento cuantos parches por WSI son analizadas en el test
# input: DataLoader
# ouput: Dict con los nombres de las WSI y su cantidad de parches
def dict_patches(data_loader):

    list_dict = [] # Crep una lista donde iré guardando los nombres

    for data in enumerate(data_loader): # Recorremos el data loader
        # data[0] = num de parche; int
        # data[1] = info del parche; dict

        list_dict.append(data[1]["img_metas"].data[0][0]["filename"])

    dict_numpatches = {} # Creo un diccionario vacio

    for elemento in list_dict: # Por cada nombre de la lista de nombres
        if elemento[:33] in dict_numpatches: # Acorto su nombre
            dict_numpatches[elemento[:33]] += 1 # aumento la ctda
        else:
            dict_numpatches[elemento[:33]] = 1 

    return dict_numpatches # Retorno el diccionario: {"WSI_1": 234, "WSI_2": 256, ... "WSI_X: XX"}

# Calculo la cantidad de votos que tiene en cada calificación; fish e ihc
# input: los resultados obtenidos del test y dict con la ctdad de parches por wsi (dict_numpatches)
# output: voto por cada WSI obtenido tanto para fish (0,1) e ihc (0,1,2,3)
def calculoEtiqueta(outputs, dict_numpatches):
    # Definición de varibles

    cont_fish_0 = 0 # Acumulador de fish 0
    cont_fish_1 = 0 # Acumulador de fish 1

    cont_ihc_0 = 0 # Acumulador de ihc 0
    cont_ihc_1 = 0 # Acumulador de ihc 1
    cont_ihc_2 = 0 # Acumulador de ihc 2
    cont_ihc_3 = 0 # Acumulador de ihc 3

    # Lista de ejemplo
    listaNumpatches = list(dict_numpatches.values())

    # Multiplico x2 cada elemento debido a que cada parche tiene calificación fish e ihc
    for i in range(len(listaNumpatches)):
        listaNumpatches[i] *= 2

    # Obtener el acumulado de la lista
    listaNumpatches = list(accumulate(listaNumpatches))

    list_outputs = [] # Creo lista vacia donde iré guardando los resultados
    # Se recorre todo el resultado
    for i in range(len(outputs)):
        if i % 2 == 0:  # si el índice es par, es decir la calificación fish

            pre_fish = outputs[i].tolist() # convierte el tensor a una lista de Python
            lista_fish = pre_fish[0] # Accede a la lista

            max_fish = max(lista_fish) # Saca el maximo
            posicion_fish = lista_fish.index(max_fish) # Determina en que posición. POS 0 = fish 0, POS 1 = fish 1

            if posicion_fish == 1: 
                cont_fish_1 = cont_fish_1 + 1
            else:
                cont_fish_0 = cont_fish_0 + 1
            
        else:  # si el índice es impar, es decir, la calificación ihc
            
            pre_ihc = outputs[i].tolist() # convierte el tensor a una lista de Python
            lista_ihc = pre_ihc[0] # Accede a la lista

            max_ihc = max(lista_ihc) # Saca el maximo
            posicion_ihc = lista_ihc.index(max_ihc) # Determina en que posición. POS 0 = ihc 0, POS 1 = ihc 1, POS 2 = ihc 2, POS 3 = ihc 3

            if posicion_ihc == 0:
                cont_ihc_0 = cont_ihc_0 + 1
            elif posicion_ihc == 1:
                cont_ihc_1 = cont_ihc_1 + 1
            elif posicion_ihc == 2:
                cont_ihc_2 = cont_ihc_2 + 1
            else:
                cont_ihc_3 = cont_ihc_3 + 1

        for j in range(len(listaNumpatches)):
            if listaNumpatches[j]-1  == i: # gracias a la lista acumulada, puedo saber hasta que punto de los outputs entregados por el modelo corresponde a cada WSI

                cont_fish = [cont_fish_0, cont_fish_1] # Agrego el contador fish por cada voto obtenido
                cont_ihc = [cont_ihc_0, cont_ihc_1, cont_ihc_2, cont_ihc_3] # Agrego el contador ihc por cada voto obtenido
                list_outputs.append([cont_fish,cont_ihc]) # Agrego ambos resultados en una sola lista

                # Reiniciamos variables
                cont_fish_0 = 0 # Acumulador de fish 0
                cont_fish_1 = 0 # Acumulador de fish 1

                cont_ihc_0 = 0 # Acumulador de ihc 0
                cont_ihc_1 = 0 # Acumulador de ihc 1
                cont_ihc_2 = 0 # Acumulador de ihc 2
                cont_ihc_3 = 0 # Acumulador de ihc 3

    return list_outputs # [[[123 4], [110 14 3 0], ...]

# Determinar la calificación final obtenida por el modelo
# input: list_ouputs y dict_numpatches
# output: Lista con el nombre y la calificación ihc y fish de cada WSI
def calificacion_final(list_ouputs,dict_numpatches):

    finalOutput = [] # Defino una lista vacia donde guardaré los resultados finales

    for element in list_ouputs: # Por cada elemento capturado en list_outputs 
        
        max_fish = max(element[0]) # Saca el maximo de fish
        max_ihc = max(element[1]) # Saca el maximo de ihc
        posicion_fish = element[0].index(max_fish) # Determina en que posición. POS 0 = fish 0, POS 1 = fish 1
        posicion_ihc = element[1].index(max_ihc) # Determina en que posición. POS 0 = ihc 0, POS 1 = ihc 1, POS 2 = ihc 2, POS 3 = ihc 3

        # Reviso el fish
        if posicion_fish == 1:
            fish_output = 1
        else:
            fish_output = 0

        # Reviso el ihc
        if posicion_ihc == 0:
            ihc_output = 0
        elif posicion_ihc == 1:
            ihc_output = 1
        elif posicion_ihc == 2:
            ihc_output = 2
        else:
            ihc_output = 3

        finalOutput.append([ihc_output, fish_output]) # Agrego el resultado con mayor votación

    lista_name_slide = list(dict_numpatches.keys()) # obtengo los nombres de cada WSI analizada

    finalOutPutReal = []
    i = 0
    while i < len(finalOutput):
        listaAux = []
        listaAux.append(lista_name_slide[i][-7:]) # Agrego a listAux el nombre
        listaAux.append(finalOutput[i][0]) # Agrego a listAux el ihc
        listaAux.append(finalOutput[i][1]) # Agrego a listAux el fish
        finalOutPutReal.append(listaAux) # Agrego la listaAux
        i = i + 1

    return finalOutPutReal # [[name, ihc, fish],[name, ihc, fish], [name, ihc, fish], ... [name, ihc, fish]]

# Función que determina la calificación original de la WSI analizada
# input: dict_numpatches
# outpu: lista con el score original
def calificacion_original(dict_numpatches, num_test):

    listaSlide = list(dict_numpatches.keys()) # Capturo los nombres de los wsi 

    listNameSlide = [] # Para guardar sólo el nombre del wsi y no la ruta completa
    for i in range(len(listaSlide)):
        nameSlide = listaSlide[i][-7:] # Acorto el string
        listNameSlide.append(nameSlide) # Guardo el nombre

    list_score_original = []  # diccionario para almacenar los valores únicos encontrados

    with open('HER2Classification/Test_'+str(num_test)+'/test_'+str(num_test)+'.csv', newline='') as archivo: # Abro el archivo csv usado para entrenar
        lector_csv = csv.DictReader(archivo) # variable para iterar por fila
        for variable in listNameSlide: # por cada nombre
            for fila in lector_csv: # por cada fila
                if variable == fila["slide_bn"]: # si el nombre es igual al nombre decrito en el archivo csv
                    if variable not in list_score_original: # agregara la calificación original sólo si no esta ya en la lista
                        list_score_original.append(fila['slide_bn']) # Agrego el nombre
                        list_score_original.append(int(fila['ihc'])) # Agrego la calificación ihc
                        list_score_original.append(int(fila['fish'])) # Agrego la calificación fish
            archivo.seek(0)  # volver al inicio del archivo para buscar la siguiente variable

    # Inicializar una lista vacía para guardar las sublistas
    sublistas = []

    # Iterar sobre los elementos de la lista, saltando de tres en tres
    for i in range(0, len(list_score_original), 3):
        # Crear una sublista de tres elementos a partir de la posición actual del bucle
        sublista = list_score_original[i:i+3]
        # Agregar la sublista a la lista de sublistas
        sublistas.append(sublista)

    list_score_original = sublistas 

    return list_score_original # [[name, ihc, fish],[name, ihc, fish], [name, ihc, fish], ... [name, ihc, fish]]

# Funcion para determinar el acc de los resultados 
# input: score orinigal y score predecidos
# output: acc de cada calificación
def crearMatrizConfusion(list_score_original, finalOutput):

    # supongamos que estos son los valores reales y predichos de una clasificación de 4 clases
    list_true_fish = [] 
    list_pred_fish = []
    list_true_ihc = []
    list_pred_ihc = []

    i = 0
    while i < len(finalOutput): # Recorremos las listas, tanto calificación real con la determinada
        # fish
        list_true_fish.append(list_score_original[i][2]) # Agregamos la calificación original
        list_pred_fish.append(finalOutput[i][2]) # Agregamos la calificacion predecida

        #ihc
        list_true_ihc.append(list_score_original[i][1]) # Agregamos la calificación original
        list_pred_ihc.append(finalOutput[i][1]) # Agregamos la calificacion predecida

        i = i + 1

    # creamos la matriz de confusión
    #confusionFish = confusion_matrix(list_true_fish, list_pred_fish)
    #confusionIhc = confusion_matrix(list_true_ihc, list_pred_ihc)
   
    # Calculo del acc
    accFish = accuracy_score(list_true_fish, list_pred_fish)
    accIhc = accuracy_score(list_true_ihc, list_pred_ihc)

    return accFish, accIhc

if __name__ == '__main__':

    data_loader, outputs = main()
    
    # CALCULAR SALIDAS 

    list_dict = dict_patches(data_loader)
    list_original = calificacion_original(list_dict, num_test=3)
    list_outputs = calculoEtiqueta(outputs,list_dict)
    list_outputs = calificacion_final(list_outputs,list_dict)

    print("-" * 100)
    print("Calificación Original: ", list_original)
    print("-" * 100)
    print("Califiación Predecida: ", list_outputs)
    print("\n\n")

    accFish, accIhc = crearMatrizConfusion(list_original,list_outputs)
    print("Acc fish: ", accFish)
    print("Acc ihc: ", accIhc)

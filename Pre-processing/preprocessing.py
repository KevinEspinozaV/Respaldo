import os, cv2, glob, re, json, logging, concurrent.futures, copy, csv, pickle
import numpy as np
import pandas as pd

from shapely.geometry import Polygon
from openslide import OpenSlide
from PIL import Image
from utils import compute_membrane
import matplotlib.pyplot as plt

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

class WSIDataset():
    def __init__(self, slide_dir, anno_dir="Dataset/", label_fn="Dataset/label.xlsx",
                 mode='train'):
        self._slide_dir, self._anno_dir = slide_dir, anno_dir
        self._label_fn, self._mode = label_fn, mode
        self._slides = self._load_slide()
        self._annos = self._load_anno()
        self._lbl_infos = self._load_label()
        self._data_infos = self._build_data_info()
        self._data_idx = 0

    def _load_slide(self):
        # load all WSI from slide_dir
        if not os.path.exists(self._slide_dir):
            return []
        slides = glob.glob("{}/**/*.mrxs".format(self._slide_dir), recursive=True)
        return slides

    def _load_anno(self):
        # load all annotations from anno_dir
        # annotation of each slide has the same basename with the WSI.
        if not os.path.exists(self._anno_dir):
            return {}
        all_annos = {}
        anno_fns = glob.glob("{}/**/*.geojson".format(self._anno_dir), recursive=True)
        logger.info("anno_fns: {}".format(anno_fns))

        for anno_fn in anno_fns:
            anno_bn = os.path.basename(anno_fn)
            anno_bn, _ = os.path.splitext(anno_bn)
            try:
                with open(anno_fn) as f:
                    anno = json.load(f)
                all_annos[anno_bn] = anno
            except Exception as e:
                logger.warning("fail to load {}. error: {}".format(anno_fn, e))
        return all_annos

    def _load_label(self):
        lbl_infos = {}
        label_df = pd.read_excel(self._label_fn)
        for row_idx, row in label_df.iterrows():
            lbl_infos[row['slide_bn']] = {'ihc': row['ihc'], 'fish': row['fish']}
        return lbl_infos

    def _correct_annotation_shift(self, annos, slide_fn):
        slide = OpenSlide(slide_fn)
        bounds_x, bounds_y = float(slide.properties['openslide.bounds-x']), float(slide.properties['openslide.bounds-y'])
        bounds = np.array([[bounds_x, bounds_y]])
        for anno in annos:
            properties, geometry = anno.get("properties", {}), anno.get("geometry", {})
            geometry_type, coordinates = geometry.get("type", ""), geometry.get("coordinates")
            if geometry_type == 'LineString':
                coordinates = np.array(coordinates) + bounds
                geometry['coordinates'] = coordinates.astype(np.int32).tolist()
            else:
                for poly_idx, poly in enumerate(coordinates):
                    poly = np.array(poly) + bounds
                    coordinates[poly_idx] = poly.astype(np.int32).tolist()
        return annos

    def _build_data_info(self):
        data_infos = []
        for slide_fn in self._slides:
            slide_bn = os.path.basename(slide_fn) # Caputra el nombre de la imagen
            slide_bn = os.path.splitext(slide_bn)[0] # Le quita la extensión .mrxs
            sample = {'slide_fn': slide_fn, 
                      'slide_bn': slide_bn,
                      'annos': self._annos.get(slide_bn, {}).get("features", []),
                      'lbls': self._lbl_infos.get(slide_bn, {})} # Crea un diccionario con los atributos mencionados anteriormente
            
            sample['annos'] = self._correct_annotation_shift(annos=sample['annos'], slide_fn=sample['slide_fn'])
            
            if self._mode == 'train':
                if len(sample['lbls']) > 0:
                    data_infos.append(sample)
                else:
                    logger.warning("training slide: {} doesn't have label.".format(slide_fn))
            else:
                data_infos.append(sample)
        return data_infos

    def __len__(self):
        return len(self._data_infos)

    def __iter__(self):
        self._data_idx = 0
        return self

    def __next__(self):
        if self._data_idx >= len(self._data_infos):
            raise StopIteration
        retval = {key: copy.deepcopy(val) for key, val in self._data_infos[self._data_idx].items()}
        retval['slide'] = OpenSlide(retval['slide_fn'])
        self._data_idx += 1
        return retval


def slide_to_patches(data,
                     crop_level=1,
                     crop_size=512,
                     exclude_annos=["EXCLUDE"],
                     anno_thres=0.1,
                     keep_patch_image=True,
                     use_cache=True,
                     output_dir=".slide_patch_cache",
                     stride=1,
                     **kwargs):
    def _poly_iou(rect, poly):
        xmin, ymin, xmax, ymax = rect
        poly1 = Polygon([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        poly2 = Polygon(poly)
        intersect = poly1.intersection(poly2)
        if intersect is None:
            return 0.0
        iou = intersect.area/min(poly1.area, poly2.area)
        return iou

    def _iou_with_annos(patch_rect):
        iou, anno_polys = [], []
        for anno in annos:
            properties, geometry = anno.get("properties", {}), anno.get("geometry", {})
            classification = properties.get("classification", {}).get("name", "###")
            if classification == "###":
                # try to directly get name from properties
                classification = properties.get("name", "###")
            geometry_type, coordinates = geometry.get("type", ""), geometry.get("coordinates")

            if classification not in exclude_annos:
                continue
            if geometry_type != 'Polygon':
                logger.warning("slide {}: unknown anno geometry_type {}".format(slide_bn, geometry_type))
                continue
            if len(coordinates) > 1:
                logger.warning("slide {}: anno geometry has more than 1 polygon. (#{})".format(slide_bn, len(coordinates)))
                continue
            polygon = coordinates[0]
            iou.append(_poly_iou(rect=patch_rect, poly=polygon))
            anno_polys.append(polygon)
        return iou, anno_polys

    # Funcion que devuelve los resumen del parche
    # Return: lista de parches [{img1},{img2}]
    def _load_cache(cache_dir):
        patches = []
        patch_fns = glob.glob("{}/{}_**.png".format(cache_dir, crop_level))
        for patch_fn in patch_fns:
            patch_bn_pattern = "(?P<crop_level>[\d.]+)_(?P<idx>[\d.]+)_(?P<idy>[\d.]+)_(?P<crop_size>[\d.]+).png"
            m = re.search(patch_bn_pattern, os.path.basename(patch_fn))
            if m is None: continue
            patch_info = {"patch_name": patch_fn,
                          "crop_level": int(m.group("crop_level")),
                          "idx": int(m.group("idx")),
                          "idy": int(m.group("idy")),
                          "crop_size": int(m.group("crop_size"))}
            if keep_patch_image:
                patch = cv2.imread(patch_fn)
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch_info['patch'] = patch
                patches.append(patch_info)

        return patches

    slide_bn, slide_name, annos = data['slide_bn'], data['slide_fn'], data['annos'] # Agregado, orginial: "slide_bn, slide, annos = data['slide_bn'], data['slide'], data['annos'] "
    patches, cache_dir = [], ""
    if use_cache:
        cache_dir = os.path.join(output_dir, slide_bn)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        patches = _load_cache(cache_dir)
        if len(patches) > 0:
            data['patches'] = patches
            return data
        
    slide = OpenSlide(slide_name) # Agregado

    width, height = slide.dimensions
    pixel_ratio = 2**crop_level
    thumbnail = slide.get_thumbnail(slide.level_dimensions[6])
    thumbnail = np.array(thumbnail)

    bounds_x, bounds_y = int(slide.properties['openslide.bounds-x']), int(slide.properties['openslide.bounds-y'])
    bounds_width, bound_height = int(slide.properties['openslide.bounds-width']), int(slide.properties['openslide.bounds-height'])

    for idx in range(bounds_x, bounds_width, crop_size * pixel_ratio):
        if idx % stride != 0: continue
        for idy in range(bounds_y, bound_height, crop_size * pixel_ratio):
            if idy % stride != 0: continue
            location, size = (idx, idy), (crop_size, crop_size)
            # use thumbnail image to check whether this patch belongs to background (blank region).
            x0, y0 = max(0, idx//64-1), max(0, idy//64-1)
            x1, y1 = idx + crop_size * pixel_ratio, idy + crop_size * pixel_ratio
            x1, y1 = min(thumbnail.shape[1], x1//64 + 1), min(thumbnail.shape[0], y1//64 + 1)
            if np.min(thumbnail[y0:y1, x0:x1]) == 255:
                continue

            # compute patch intersection with excluded annotation region
            patch_rect = [idx, idy, idx+crop_size*pixel_ratio, idy+crop_size*pixel_ratio]
            iou, anno_polys = _iou_with_annos(patch_rect)
            # ignore patches with iou with excluded regions over threshold
            if any([x >= anno_thres for x in iou]):
                continue

            patch = slide.read_region(location=location,
                                      level=crop_level,
                                      size=size)
            patch = np.array(patch.convert('RGB'))
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            # ignore patches closed to the boundary of valid region
            if np.sum(patch_gray == 0)/(patch_gray.shape[0]*patch_gray.shape[1]) > 0.05:
                continue
            # ignore patches with too large invisible area
            if np.sum(patch_gray > 245)/(patch_gray.shape[0]*patch_gray.shape[1]) > 0.9:
                continue

            for poly_idx, (iou_val, anno_poly) in enumerate(zip(iou, anno_polys)):
                if iou_val > 0:
                    # mask annotated region from the patch
                    anno_poly = np.array(anno_poly) - np.array([[idx, idy]])
                    anno_poly = anno_poly / pixel_ratio
                    cv2.fillPoly(patch, [anno_poly.astype(np.int32)], [0, 0, 0])

            patch_info = {"patch_fn": "",
                          "crop_level": crop_level,
                          "idx": idx,
                          "idy": idy,
                          "crop_size": crop_size,
                          "slide_bn": slide_bn,
                          "patch": None}

            if use_cache:
                patch_fn = os.path.join(cache_dir, "{}_{}_{}_{}.png".format(crop_level, idx, idy, crop_size))
                patch_info['patch_fn'] = patch_fn
                Image.fromarray(patch).save(patch_fn)

            if keep_patch_image:
                patch_info['patch'] = patch
            patches.append(patch_info)

    data['patches'] = patches
    return data

def compute_graymap(data, max_workers=8, keep_membrane_mask=False, **kwargs):
    # compute gray value for each patch
    patches = data.get("patches", [])
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as excutor:
        future_to_patch_idx = {excutor.submit(compute_membrane, patch['patch'], **kwargs): patch_idx for patch_idx, patch in enumerate(patches)}
        for future in concurrent.futures.as_completed(future_to_patch_idx):
            patch_idx = future_to_patch_idx[future]
            try:
                membrane_mask, (gray_val, membrane_pixels_frac, lightness) = future.result()
                if keep_membrane_mask:
                    patches[patch_idx]['membrane_mask'] = membrane_mask
                patches[patch_idx]['gray_val'] = gray_val
                patches[patch_idx]['membrane_pixels_frac'] = membrane_pixels_frac
                patches[patch_idx]['lightness'] = lightness
            except Exception as exc:
                logger.error("patch: {} generates an exception： {}".format(patch_idx, exc))
    # compute graymap
    slide_name = data['slide_fn'] # toma el nombre
    slide = OpenSlide(slide_name)
    patch_crop_level, patch_width, patch_height = patches[0]["crop_level"], patches[0]['patch'].shape[1], patches[0]['patch'].shape[0]
    width, height = slide.level_dimensions[patch_crop_level]
    graymap_size = int(np.ceil(width/patch_width)), int(np.ceil(height/patch_height))
    graymap = np.zeros((graymap_size[1], graymap_size[0], 3), dtype=np.float32)
    for patch in patches:
        idx, idy = int(patch['idx']/(2**patch_crop_level*patch_width)), int(patch['idy']/(2**patch_crop_level*patch_height))
        graymap[idy, idx, 0] = patch['gray_val']
        graymap[idy, idx, 1] = patch['membrane_pixels_frac']
        graymap[idy, idx, 2] = patch['lightness']

        '''# Crear un archivo pickle
        with open(".pkl"), 'wb') as f:
            pickle.dump(graymap["graymap"], f)'''

    data['graymap'] = graymap
    return data


def plot_data(data, plot_patch, plot_graymap, slide_display_level=4, figsize=10): # plot_patch=False, plot_graymap=False,
    def _plot_patch(data):
        patches = data['patches']

        slide_name = data['slide_fn']
        slide = OpenSlide(slide_name)

        width, height = slide.level_dimensions[slide_display_level]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        patch_crop_level = patches[0]["crop_level"]
        for patch in patches:
            idx, idy, patch = patch['idx'], patch['idy'], patch.get("patch")

            # patch坐标
            idx0, idy0 = idx // 2**slide_display_level, idy // 2**slide_display_level
            display_width, display_height = patch.shape[1] // 2**(slide_display_level - patch_crop_level), \
                                            patch.shape[0] // 2**(slide_display_level - patch_crop_level)
            patch = cv2.resize(patch, (int(display_width), int(display_height)))
            idx1, idy1 = min(width, idx0 + patch.shape[1]), min(height, idy0 + patch.shape[0])
            canvas[idy0:idy1, idx0:idx1] = patch[:idy1 - idy0, :idx1 - idx0]

        thumbnail = np.array(slide.get_thumbnail((width, height)).convert('RGB'))
        for anno in data['annos']:
            coordinates = anno['geometry']['coordinates'][0]
            coordinates = np.array(coordinates)/2**slide_display_level
            cv2.polylines(thumbnail, [coordinates.astype(np.int32)], 1, [255, 0, 0], 10)

        plt.figure(figsize=(figsize, 2*figsize))
        plt.subplot(1, 2, 1)
        plt.imshow(canvas)
        plt.title("patch images")
        plt.subplot(1, 2, 2)
        plt.imshow(thumbnail)
        plt.title("reference")
        plt.show()

    if plot_patch:
        _plot_patch(data)

    if plot_graymap:
        graymap = data['graymap']
        titles = ['gray_val', 'membrane_pixels_frac', 'lightness']
        plt.figure(figsize=(figsize, 3*figsize))
        for idx in range(3):
            plt.subplot(1, 3, idx+1)
            plt.imshow(graymap[...,idx])
            plt.title(titles[idx])
        plt.show()

data = WSIDataset(slide_dir='Dataset')
patches = slide_to_patches(data._data_infos[0])
graymap = compute_graymap(patches)
plot_data(graymap,False,True)

# Crear un archivo pickle
nombreArchivo = data._data_infos[0]["slide_bn"]
with open(nombreArchivo+str(".pkl"), 'wb') as f:
    pickle.dump(graymap["graymap"], f)

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from click import Argument

import mmcv
from PIL import Image
import numpy as np
from shapely.geometry import Polygon

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert images to coco format without annotations')
    parser.add_argument('img_path', help='The root path of images')
    parser.add_argument(
        'classes', type=str, help='The text file name of storage class list')
    parser.add_argument(
        'out',
        type=str,
        help='The output annotation json file name, The save dir is in the '
        'same directory as img_path')
    parser.add_argument(
        '-e',
        '--exclude-extensions',
        type=str,
        nargs='+',
        help='The suffix of images to be excluded, such as "png" and "bmp"')
    args = parser.parse_args()
    return args


def collect_image_infos(path, exclude_extensions=None):
    img_infos = []

    images_generator = mmcv.scandir(path, recursive=True)
    for image_path in mmcv.track_iter_progress(list(images_generator)):
        if exclude_extensions is None or (
                exclude_extensions is not None
                and not image_path.lower().endswith(exclude_extensions)):
            image_path = os.path.join(path, image_path)
            img_pillow = Image.open(image_path)
            img_info = {
                'filename': image_path,
                'width': img_pillow.width,
                'height': img_pillow.height,
            }
            img_infos.append(img_info)
    return img_infos

def collect_image_infos_v2(path, exclude_extensions=None):
    img_infos = []

    images_generator = mmcv.scandir(path, recursive=True)
    for image_path in mmcv.track_iter_progress(list(images_generator)):
        if exclude_extensions is None or (
                exclude_extensions is not None
                and not image_path.lower().endswith(exclude_extensions)):
            image_path = os.path.join(path, image_path)
            img_pillow = Image.open(image_path)
            img_info = {
                'filename': image_path,
                'width': img_pillow.width,
                'height': img_pillow.height,
            }
            img_infos.append(img_info)
    return img_infos

def cvt_to_coco_json(json_infos, classes):
    image_id = 1
    annotations_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    for category_id, name in enumerate(classes):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id) + 1
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    error_1, error_2 = 0, 0
    for img_dict in json_infos:
        file_name = img_dict['image_name']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        try:
            image_item['height'] = int(img_dict['image_height'])
            image_item['width'] = int(img_dict['image_width'])
        except KeyError:
            W, H = Image.open(f"/data/liuzhuang/Dataset/text_det/images/train/{file_name}").size
            image_item['height'] = H
            image_item['width'] = W
        except IOError:
            error_1 += 1
            continue
        
        image_set.add(file_name)
        image_info = []
        for index, ann_dict in enumerate(img_dict['text']):
            try:
                area = Polygon(ann_dict['vertices']).area
                bbox = ann_dict['bbox']
                segmentation = [bbox_flatten(ann_dict['vertices']),]
                image_info.append({
                    "id": annotations_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "segmentation": segmentation,
                    "area": area,
                    "transcription": "PETROSAINS",
                    "iscrowd": 0
                })
            except KeyError:
                area = Polygon(ann_dict['vertices']).area
                bbox = polygon2bbox(ann_dict['vertices'])
                segmentation = [bbox_flatten(ann_dict['vertices']),]
                image_info.append({
                    "id": annotations_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "segmentation": segmentation,
                    "area": area,
                    "transcription": "PETROSAINS",
                    "iscrowd": 0
                })
            except Exception:
                annotations_id -= index
                error_2 += 1
                break
            annotations_id += 1
        else:            
            image_id += 1
            coco['images'].append(image_item)
            coco['annotations'] += image_info
    print("Image info error:", error_1)
    print("Image bbox info error:", error_2)
    return coco

def polygon2bbox(bbox):
    b = np.array(bbox)
    b_x, b_y, t_x, t_y = b[:, 0].max(), b[:, 1].max(), b[:, 0].min(), b[:, 1].min()
    return np.array([t_x, t_y, b_x-t_x, b_y-t_y]).tolist()

def bbox_flatten(bbox):
    b = np.array(bbox)
    b = b.flatten()
    return b.tolist()

def main():
    # 1 load image list info
    json_infos = json.load(open(args.json_path))

    # 2 convert to coco format data
    classes = ('text', )
    coco_info = cvt_to_coco_json(json_infos, classes)
    json.dump(coco_info, open(args.json_path+"_coco.json", 'w'), ensure_ascii=False)

if __name__ == '__main__':
    import json
    
    parser = argparse.ArgumentParser(description='Convert images to coco format without annotations')
    parser.add_argument('-j', '--json_path', default='/data/liuzhuang/Dataset/text_det/annos/tie_val_v1.json', help='The root path of images')
    args = parser.parse_args()
    main()

import cv2
import numpy as np

from mmdet.apis.inference import inference_detector, init_detector

model = init_detector(config='work_dirs/cascade_mask_rcnn_r50_fpn_20e_coco/cascade_mask_rcnn_r50_fpn_20e_coco.py', )
results = inference_detector(model, imgs=['/data/liuzhuang/Dataset/text_det/images/train/openimages_v5/train_5/58a3acaf54b3f170.jpg'])
for result in results:        
    bbox, segm = result
    for i, j in zip(bbox, segm):
        for b, s in zip(i, j):
            contours, hierarchy = cv2.findContours(s.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            rect = cv2.minAreaRect(cnt)
            ploygon = cv2.boxPoints(rect)
            ploygon = np.int0(ploygon)
            print(ploygon)

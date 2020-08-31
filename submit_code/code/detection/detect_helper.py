import os
import time
from tqdm import tqdm

import cv2
import numpy as np
import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
from torch.utils.data import Dataset, DataLoader
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, get_index_label, plot_one_box


def count_objects(preds, obj_list, res, imshow=True, imwrite=False):
    """
    数结果中有多少objects, 排除<50的目标
    """
    for i in range(len(preds)):
        person_nonvehicle_num = 0
        vehicle_num = 0
        if len(preds[i]['rois']) == 0:
            res["person_nonvehicle_num"].append(0)
            res["vehicle_num"].append(0)
            res["max_area"].append(0)
            continue
        #人, 非机动车，机动车
        #0, 1+3, 2+5+7
        # 画出裁剪矩形
        max_area = 0
        for j in range(len(preds[i]['rois'])):
            col1, row1, col2, row2 = preds[i]['rois'][j].astype(np.int)
            width = col2 - col1
            height = row2 - row1
            if width < 30 or height < 30:
                continue
            if width/height > 3.5:
                continue
            if width > 800:
                continue
            if width * height > max_area:
                max_area = width * height
            pred_class = preds[i]['class_ids'][j]
            if pred_class == 0:
                if height/width>1:
                    person_nonvehicle_num += 1
            if pred_class in [1,3]:
                person_nonvehicle_num += 1
            if pred_class in [2,5,7]:
                vehicle_num += 1
            obj = obj_list[pred_class]
            score = float(preds[i]['scores'][j])
            # obj.
        res["person_nonvehicle_num"].append(person_nonvehicle_num)
        res["vehicle_num"].append(vehicle_num)
        res["max_area"].append(max_area)


def detect(img_path_list):
    """
    input:
        img_path_list
    return:
        res dataframe： person_nonvehicle_num, vehicle_num
    """
    compound_coef = 4
    force_input_size = None  # set None to use default size
     # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    threshold = 0.2
    iou_threshold = 0.2
    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True
    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    batch_size = 8


    # model
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'), strict=False)
    model = model.cuda()
    model.requires_grad_(False)
    model.eval()
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    res = {}
    res["person_nonvehicle_num"] = []
    res["vehicle_num"] = []
    res["max_area"] = []
    count = 0
    for i in tqdm(range(0, len(img_path_list), batch_size)):
        ori_imgs, framed_imgs, framed_metas = preprocess(img_path_list[i:i+batch_size], max_size=input_size)
        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
        x = x.to(torch.float32).permute(0, 3, 1, 2)
        with torch.no_grad():
            _, regression, classification, anchors = model(x)

            out = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, iou_threshold)
            # out = invert_affine(framed_metas, out)
            count += len(out)
        count_objects(out, obj_list, res, imshow=False, imwrite=True)
        pass
    from pandas.core.frame import DataFrame
    res = DataFrame(res)
    return res
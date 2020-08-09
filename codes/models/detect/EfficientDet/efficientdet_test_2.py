# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import os
import time
import torch
from torch.backends import cudnn
from matplotlib import colors
from backbone import EfficientDetBackbone
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
from tqdm import tqdm

import sys
sys.path.append("/workdir/congest/codes") # 这句是为了导入_config
from common.split_data import split_train_val 

def view_json(json_object, data_root, train_valid):
    """
    input:
        train_valid: 0-train, 1-valid
    """
    # view the data
    data_array = []
    for frame in json_object["frames"]:
        data_dict = {}
        data_dict["train_valid"] = train_valid
        data_dict["id"] = json_object["id"]
        data_dict["status"] = json_object["status"]
        data_dict["key_frame"] = json_object["key_frame"]
        data_dict["frame_name"] = frame["frame_name"]
        data_dict["gps_time"] = frame["gps_time"]
        data_dict["path"] = os.path.join(data_root, data_dict["id"], data_dict["frame_name"])
        data_array.append(data_dict)
    return data_array


def count_objects(preds, obj_list, res, imshow=True, imwrite=False):
    """
    数结果中有多少objects
    """
    for i in range(len(preds)):
        person_num = 0
        nonvehicle_num = 0
        vehicle_num = 0
        if len(preds[i]['rois']) == 0:
            res["person_num"].append(0)
            res["nonvehicle_num"].append(0)
            res["vehicle_num"].append(0)
            res["max_area"].append(0)
            continue
        #人, 非机动车，机动车
        #0, 1+3, 2+5+7
        # 画出裁剪矩形
        max_area = 0
        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            if (x2-x1) * (y2-y1) > max_area:
                max_area = (x2-x1) * (y2-y1)
            pred_class = preds[i]['class_ids'][j]
            if pred_class == 0:
                person_num += 1
            if pred_class in [1,3]:
                nonvehicle_num += 1
            if pred_class in [2,5,7]:
                vehicle_num += 1
            obj = obj_list[pred_class]
            score = float(preds[i]['scores'][j])
            # obj.
        res["person_num"].append(person_num)
        res["nonvehicle_num"].append(nonvehicle_num)
        res["vehicle_num"].append(vehicle_num)
        res["max_area"].append(max_area)
    # return res


def detect(img_path_list):
    """
    input:
        img_path_list
    return:
        res dataframe： person_num, nonvehicle_num, vehicle_num
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

    color_list = standard_to_bgr(STANDARD_COLORS)
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    batch_size=12


    # model
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
    model = model.cuda()
    model.requires_grad_(False)
    model.eval()
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    res = {}
    res["person_num"] = []
    res["nonvehicle_num"] = []
    res["vehicle_num"] = []
    res["max_area"] = []
    count = 0
    for i in tqdm(range(0, len(img_path_list), batch_size)):
        ori_imgs, framed_imgs, framed_metas = preprocess(img_path_list[i:i+batch_size], max_size=input_size)
        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
        with torch.no_grad():
            _, regression, classification, anchors = model(x)

            out = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, iou_threshold)
            # out = invert_affine(framed_metas, out)
            count += len(out)
        count_objects(out, obj_list, res, imshow=False, imwrite=True)
    from pandas.core.frame import DataFrame
    res = DataFrame(res)
    return res


if __name__ == "__main__":
    json_path = "/workdir/congest/datasets/amap_traffic_annotations_train.json"
    imgs_root = "/workdir/congest/datasets/amap_traffic_train_0712"
    train, val = split_train_val(json_path, imgs_root)
    print("train", len(train[0]),len(train[1]),len(train[2]))
    print("valid", len(val[0]),len(val[1]),len(val[2]))
    df_data = []
    train_valid = 0
    for data in train[0]:
        df_data.extend(view_json(data, imgs_root, train_valid))
    for data in train[1]:
        df_data.extend(view_json(data, imgs_root, train_valid))
    for data in train[2]:
        df_data.extend(view_json(data, imgs_root, train_valid))
    train_valid = 1
    for data in val[0]:
        df_data.extend(view_json(data, imgs_root, train_valid))
    for data in val[1]:
        df_data.extend(view_json(data, imgs_root, train_valid))
    for data in val[2]:
        df_data.extend(view_json(data, imgs_root, train_valid))
    
    from pandas.core.frame import DataFrame
    import pandas as pd
    df_data = DataFrame(df_data)
    # df_data = df_data.head(len(df_data)//2)
    df_data = df_data.tail(len(df_data) - len(df_data)//2)
    df_data.gps_time = pd.to_datetime(df_data.gps_time.values, unit='s', utc=True).tz_convert("Asia/Shanghai")
    # df_train.gps_time = pd.to_datetime(df_train.gps_time.values, unit='s', utc=False)
    df_data["is_keyframe"] = (df_data["key_frame"]==df_data["frame_name"]).astype('int')
    img_path_list = np.array(df_data['path'])#np.ndarray()
    img_path_list = img_path_list.tolist()#list
    detect_res = detect(img_path_list)
    detect_res.reset_index(drop=True, inplace=True)
    df_data.reset_index(drop=True, inplace=True)
    df_data = pd.concat([df_data, detect_res],axis=1)
    # print(df_data)
    df_data.to_pickle("detection_2.pkl")

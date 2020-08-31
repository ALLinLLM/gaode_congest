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
from detect_helper import detect


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


if __name__ == "__main__":
    json_path = "../../data/amap_traffic_annotations_b_test_0828.json"
    imgs_root = "../../data/amap_traffic_b_test_0828"

    import json
    with open(json_path, "r", encoding="utf-8") as f:
        json_dict = json.load(f)
        data_arr = json_dict["annotations"]
    
    #---- DEBUG
    # data_arr = data_arr[:10]
    #---- DEBUG

    print("test data size:", len(data_arr))
    df_data = []
    train_valid = 2
    for data in data_arr:
        df_data.extend(view_json(data, imgs_root, train_valid))
    
    print("test data frame numbers:", len(df_data))
    from pandas.core.frame import DataFrame
    import pandas as pd
    df_data = DataFrame(df_data)
    # df_data = df_data.head(len(df_data)//2)
    # df_data2 = df_data.tail(len(df_data) - len(df_data)//2)
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
    df_data.to_pickle("../../user_data/tmp_data/detection_testb.pkl")

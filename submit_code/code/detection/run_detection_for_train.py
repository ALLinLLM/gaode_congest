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
import sys
sys.path.append("..") # 这句是为了导入_config
from data.json_helper import split_train_val

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


if __name__ == "__main__":
    json_path = "../../data/amap_traffic_annotations_train.json"
    imgs_root = "../../data/amap_traffic_train_0712"
    train, val = split_train_val(json_path, imgs_root)
    print("train", len(train[0]),len(train[1]),len(train[2]))
    print("valid", len(val[0]),len(val[1]),len(val[2]))
    df_data = []
    train_valid = 0
    # label: 0, 1, 2
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
    
    #---- DEBUG
    # df_data = df_data[:10]
    #---- DEBUG

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
    df_data.to_pickle("../../user_data/tmp_data/detection_train.pkl")

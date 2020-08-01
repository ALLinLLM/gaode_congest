import json
import numpy as np
import random


def get_data(imgs_root, data):
    return imgs_root + "/" + data["frames"][0]["frame_name"], imgs_root + "/" + data["frames"][-1]["frame_name"]

def get_trainXy_validXy(total_num=-1, valid_num=100):
    """
    同一序列的5帧作为一个样本, 分别输入VGG19提取特征后, 用来计算首尾两帧的帧间cos相似度
    """
    json_path = "/workdir/tianchi/gaode_congest/datasets/amap_traffic_annotations_train.json"
    imgs_root = "/workdir/tianchi/gaode_congest/datasets/amap_traffic_train_0712"
    valid_pass_num = valid_num*0.7//1
    current_valid_pass_num = 0 
    valid_slow_num = valid_num*0.1//1
    current_valid_slow_num = 0
    valid_congest_num = valid_num*0.2//1
    current_valid_congest_num = 0
    with open(json_path, "r", encoding="utf-8") as f:
        json_dict = json.load(f)
        data_arr = json_dict["annotations"]

    random.seed(2020)
    random.shuffle(data_arr)
    if total_num>-1:
        data_arr = data_arr[:total_num]
    count = 0
    test_X_list, test_y_list = [], []
    X_list, y_list = [], []
    for data in data_arr:
        if data["status"]==0:
            if current_valid_pass_num<valid_pass_num:
                current_valid_pass_num += 1
                test_X_list.append(get_data(imgs_root + "/" + data["id"] , data))
                test_y_list.append(data["status"])
                continue
        if data["status"]==1:
            if current_valid_slow_num<valid_slow_num:
                current_valid_slow_num += 1
                test_X_list.append(get_data(imgs_root + "/" + data["id"] , data))
                test_y_list.append(data["status"])
                continue
        if data["status"]==2:
            if current_valid_congest_num<valid_congest_num:
                current_valid_congest_num += 1
                test_X_list.append(get_data(imgs_root + "/" + data["id"] , data))
                test_y_list.append(data["status"])
                continue
        X_list.append(get_data(imgs_root + "/" + data["id"] , data))
        y_list.append(data["status"])
    return X_list, y_list, test_X_list, test_y_list

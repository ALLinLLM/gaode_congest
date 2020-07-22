import json
import numpy as np
import random

json_path = "../../datasets/amap_traffic_annotations_train.json"
imgs_root = "/workdir/tianchi/gaode_congest-master/datasets/amap_traffic_train_0712"
valid_num = 600
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

count = 0
test_X_list, test_y_list = [], []
X_list, y_list = [], []
for data in data_arr:
    seq_path = imgs_root + "/" + data["id"] 
    for frame in data["frames"]:
        img_path = seq_path + "/" + frame["frame_name"]
        if data["status"]==0:
            if current_valid_pass_num<valid_pass_num:
                current_valid_pass_num += 1
                test_X_list.append(img_path)
                test_y_list.append(data["status"])
                continue
        if data["status"]==1:
            if current_valid_slow_num<valid_slow_num:
                current_valid_slow_num += 1
                test_X_list.append(img_path)
                test_y_list.append(data["status"])
                continue
        if data["status"]==2:
            if current_valid_congest_num<valid_congest_num:
                current_valid_congest_num += 1
                test_X_list.append(img_path)
                test_y_list.append(data["status"])
                continue
        X_list.append(img_path)
        y_list.append(data["status"])

with open("../data/valid_seed2020.txt", "w", encoding="utf8") as test:
    for x,y in zip(test_X_list, test_y_list):
        test.write(x)
        test.write(",")
        test.write(str(y))
        test.write("\n")
with open("../data/train_seed2020.txt", "w", encoding="utf8") as train:
    for x,y in zip(X_list, y_list):
        train.write(x)
        train.write(",")
        train.write(str(y))
        train.write("\n")
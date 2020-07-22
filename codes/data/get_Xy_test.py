import json
import numpy as np
import random

json_path = "../../datasets/amap_traffic_annotations_test.json"
imgs_root = "/workdir/tianchi/gaode_congest-master/datasets/amap_traffic_test_0712"

with open(json_path, "r", encoding="utf-8") as f:
    json_dict = json.load(f)
    data_arr = json_dict["annotations"]

count = 0
test_X_list, test_y_list = [], []
X_list, y_list = [], []
for data in data_arr:
    seq_path = imgs_root + "/" + data["id"] 
    img_path = seq_path + "/" + data["key_frame"]
    X_list.append(img_path)
    y_list.append(data["status"])

with open("../data/test.txt", "w", encoding="utf8") as train:
    for x,y in zip(X_list, y_list):
        train.write(x)
        train.write(",")
        train.write(str(y))
        train.write("\n")
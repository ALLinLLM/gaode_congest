# get all id
import json

def split_train_val(json_path, imgs_root):
    train_id={}

    with open(json_path, "r", encoding="utf-8") as f:
        json_dict = json.load(f)
        data_arr = json_dict["annotations"]

    for data in data_arr:
        if train_id.get(data["status"]) is None:
            train_id[data["status"]]=[data]
        else:
            train_id[data["status"]].append(data)

    train_len = len(train_id[0]) + len(train_id[1]) + len(train_id[2])

    valid_num = 200
    valid_open = round(valid_num*len(train_id[0])/train_len)
    valid_slow = round(valid_num*len(train_id[1])/train_len)
    valid_congest = round(valid_num*len(train_id[2])/train_len)

    valid_num = valid_open + valid_slow + valid_congest

    import random
    random.seed(2020)
    random.shuffle(train_id[0])
    random.shuffle(train_id[1])
    random.shuffle(train_id[2])
    valid_id={}
    valid_id[0] = train_id[0][:valid_open]
    valid_id[1] = train_id[1][:valid_slow]
    valid_id[2] = train_id[2][:valid_congest]
    train_id[0] = train_id[0][valid_open:]
    train_id[1] = train_id[1][valid_slow:]
    train_id[2] = train_id[2][valid_congest:]
    return train_id, valid_id

if __name__ == "__main__":
    json_path = "/workdir/congest/datasets/amap_traffic_annotations_train.json"
    imgs_root = "/workdir/congest/datasets/amap_traffic_train_0712"
    train, val = split_train_val(json_path, imgs_root)
    print("train", len(train[0]),len(train[1]),len(train[2]))
    print("valid", len(val[0]),len(val[1]),len(val[2]))
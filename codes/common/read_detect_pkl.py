import json
import numpy as np
import pandas as pd


def cal_dif(x):
    return x.max() - x.min()

def postprocess_detect():
    df1 = pd.read_pickle("/workdir/congest/result/detection_1.pkl")
    df2 = pd.read_pickle("/workdir/congest/result/detection_2.pkl")
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    df_data = pd.concat([df1, df2], axis=0)
    key_data = df_data[df_data["is_keyframe"]==1][['id', 'train_valid', 'person_num', "nonvehicle_num", "vehicle_num", "max_area"]]
    key_data.columns = ['id', 'train_valid', 'person_num_key', "nonvehicle_num_key", "vehicle_num_key", "max_area_key"]

    a = df_data[['id', 'person_num', "nonvehicle_num", "vehicle_num", "max_area"]].groupby(['id']).agg(['mean', 'std', 'max', 'min', cal_dif])
    b = a.copy(deep=True)
    b.columns = ["_".join(x) for x in b.columns.ravel()]
    c = key_data.join(b, on='id')
    return c

def postprocess_detect_test():
    # df1 = pd.read_pickle("/workdir/congest/result/detection_test_1.pkl")
    df_data = pd.read_pickle("/workdir/congest/result/detection_test_2.pkl")
    # df1.reset_index(drop=True, inplace=True)
    # df2.reset_index(drop=True, inplace=True)
    # df_data = pd.concat([df1, df2], axis=0)
    key_data = df_data[df_data["is_keyframe"]==1][['id', 'train_valid', 'person_num', "nonvehicle_num", "vehicle_num", "max_area"]]
    key_data.columns = ['id', 'train_valid', 'person_num_key', "nonvehicle_num_key", "vehicle_num_key", "max_area_key"]

    a = df_data[['id', 'person_num', "nonvehicle_num", "vehicle_num", "max_area"]].groupby(['id']).agg(['mean', 'std', 'max', 'min', cal_dif])
    b = a.copy(deep=True)
    b.columns = ["_".join(x) for x in b.columns.ravel()]
    c = key_data.join(b, on='id')
    return c

def postprocess_cos_sim():
    json_path="/workdir/congest/result/cos_anno.json"
    with open(json_path, "r", encoding="utf-8") as f:
        json_dict = json.load(f)
        df_dict = {"id":[], "cos_mean":[], "cos_std":[], "cos_max":[], "cos_min":[]}
        for k, v in json_dict.items():
            df_dict["id"].append(k)
            cos = []
            for i, j in v.items():
                if i == "status": continue
                cos.append(j["cos_sim"])
            cos = np.array(cos)
            df_dict["cos_mean"].append(cos.mean())
            df_dict["cos_std"].append(cos.std())
            df_dict["cos_max"].append(cos.max())
            df_dict["cos_min"].append(cos.min())
        from pandas.core.frame import DataFrame
        df_dict = DataFrame(df_dict)
        df_dict.to_pickle("cos_sim.pkl")

if __name__ == "__main__":
    postprocess_detect()
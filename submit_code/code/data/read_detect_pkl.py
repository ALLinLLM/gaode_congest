import json
import numpy as np
import pandas as pd


def cal_dif(x):
    return x.max() - x.min()

def postprocess_detect(df_pkl_path):
    df_data = pd.read_pickle(df_pkl_path)
    df_data.reset_index(drop=True, inplace=True)
    key_data = df_data[df_data["is_keyframe"]==1][['id', 'train_valid', 'person_nonvehicle_num', "vehicle_num", "max_area"]]
    key_data.columns = ['id', 'train_valid', 'person_nonvehicle_num_key', "vehicle_num_key", "max_area_key"]

    a = df_data[['id', "person_nonvehicle_num", "vehicle_num", "max_area"]].groupby(['id']).agg(['mean', 'std', 'max', 'min', cal_dif])
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

def postprocess_cos_sim(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        json_dict = json.load(f)
        df_dict = {"id":[], "cos_mean":[], "cos_std":[], "cos_max":[], "cos_min":[], "cos_v_mean":[], "cos_v_std":[], "cos_v_max":[], "cos_v_min":[]}
        for k, v in json_dict.items():
            df_dict["id"].append(k)
            cos = []
            cos_v = []
            for i, j in v.items():
                if i == "status": continue
                if j["interval"]==0: continue
                cos.append(j["cos_sim"])
                cos_v.append(j["cos_sim"]/j["interval"])
            cos = np.array(cos)
            cos_v = np.array(cos_v)
            df_dict["cos_mean"].append(cos.mean())
            df_dict["cos_std"].append(cos.std())
            df_dict["cos_max"].append(cos.max())
            df_dict["cos_min"].append(cos.min())
            df_dict["cos_v_mean"].append(cos_v.mean())
            df_dict["cos_v_std"].append(cos_v.std())
            df_dict["cos_v_max"].append(cos_v.max())
            df_dict["cos_v_min"].append(cos_v.min())
        from pandas.core.frame import DataFrame
        df_dict = DataFrame(df_dict)
        import os 
        output = os.path.basename(json_path)[:-5]+".pkl"
        df_dict.to_pickle(output)

if __name__ == "__main__":
    # json_path="/workdir/congest/result/cos_anno.json"
    # json_path="/workdir/congest/result/cos_anno_test.json"
    # postprocess_cos_sim(json_path)
    # df_pkl_path = "../../user_data/tmp_data/detection_testb.pkl"
    df_pkl_path = "../../user_data/tmp_data/detection_train.pkl"
    postprocess_detect(df_pkl_path)
from data.read_detect_pkl import postprocess_detect, postprocess_detect_test
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import lightgbm
import matplotlib.image as mpimg
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, KFold

def get_time_features(df,img_path):
    map_id_list=[]
    label=[]
    key_frame_list=[]
    jpg_name_1=[]
    jpg_name_2=[]
    gap_time_1=[]
    gap_time_2=[]
    im_diff_mean=[]
    im_diff_std=[]
    
    for s in list(df.annotations):
        map_id=s["id"]
        map_key=s["key_frame"]
        frames=s["frames"]
        status=s["status"]
        for i in range(0,len(frames)-1):
            f=frames[i]
            f_next=frames[i+1]
            """
            im=mpimg.imread(path+img_path+"/"+map_id+"/"+f["frame_name"])
            im_next=mpimg.imread(path+img_path+"/"+map_id+"/"+f_next["frame_name"])
            
            if im.shape==im_next.shape:
                im_diff=im-im_next
            else:
                im_diff=im
            
            im_diff_mean.append(np.mean(im_diff))
            im_diff_std.append(np.std(im_diff))
            """

            map_id_list.append(map_id)
            key_frame_list.append(map_key)
            jpg_name_1.append(f["frame_name"])
            jpg_name_2.append(f_next["frame_name"])
            gap_time_1.append(f["gps_time"])
            gap_time_2.append(f_next["gps_time"])
            label.append(status)
    train_df= pd.DataFrame({
        "map_id":map_id_list,
        "label":label,
        "key_frame":key_frame_list,
        "jpg_name_1":jpg_name_1,
        "jpg_name_2":jpg_name_2,
        "gap_time_1":gap_time_1,
        "gap_time_2":gap_time_2,
        #"im_diff_mean":im_diff_mean,
        #"im_diff_std":im_diff_std,
    })

    train_df["gap"]=train_df["gap_time_2"]-train_df["gap_time_1"]
    train_df["gap_time_today"]=train_df["gap_time_1"]%(24*3600)
    train_df["hour"]=train_df["gap_time_1"].apply(lambda x:datetime.fromtimestamp(x).hour)
    train_df["minute"]=train_df["gap_time_1"].apply(lambda x:datetime.fromtimestamp(x).minute)
    train_df["day"]=train_df["gap_time_1"].apply(lambda x:datetime.fromtimestamp(x).day)
    train_df["dayofweek"]=train_df["gap_time_1"].apply(lambda x:datetime.fromtimestamp(x).weekday())
    
    train_df["key_frame"]=train_df["key_frame"].apply(lambda x:int(x.split(".")[0]))
    
    train_df=train_df.groupby("map_id").agg({"gap":["mean","std"],
                                             "hour":["mean"],
                                             "minute":["mean"],
                                             "dayofweek":["mean"],
                                             "gap_time_today":["mean","std"],
                                             #"im_diff_mean":["mean","std"],
                                             #"im_diff_std":["mean","std"],
                                             "label":["mean"],
                                            }).reset_index()
    train_df.columns=["map_id","gap_mean","gap_std",
                      "hour_mean","minute_mean","dayofweek_mean","gap_time_today_mean","gap_time_today_std",
                      #"im_diff_mean_mean","im_diff_mean_std","im_diff_std_mean","im_diff_std_std",
                      "label"]
    train_df["label"]=train_df["label"].apply(int)
    
    return train_df

def train_lgb(train_x, train_y, test_x, class_num=1):
    predictors = list(train_x.columns)
    train_x = train_x.values
    train_y = train_y.to_numpy()
    test_x = test_x.values
    folds = 12
    seed = 2020
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros((train_x.shape[0], class_num))
    test = np.zeros((test_x.shape[0], class_num))
    test_pre = np.zeros((folds, test_x.shape[0], class_num))
    test_pre_all = np.zeros((folds, test_x.shape[0]))
    cv_scores = []
    f1_scores = []
    cv_rounds = []

    for i, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]

        train_matrix = lightgbm.Dataset(tr_x, label=tr_y)
        test_matrix = lightgbm.Dataset(te_x, label=te_y)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            #'metric': 'None',
            'metric': 'multi_logloss',
            'min_child_weight': 1.5,
            'num_leaves': 2 ** 3-1,
            'lambda_l2': 10,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 4,
            'learning_rate': 0.05,
            'seed': 2019,
            'nthread': 28,
            'num_class': class_num,
            'silent': True,
            'verbose': -1,
        }

        num_round = 4000
        early_stopping_rounds = 100
        if test_matrix:
            model = lightgbm.train(params, train_matrix, num_round, valid_sets=test_matrix, verbose_eval=50,
                                #feval=acc_score_vali,
                                early_stopping_rounds=early_stopping_rounds
                                )
            print("\n".join(("%s: %.2f" % x) for x in
                            list(sorted(zip(predictors, model.feature_importance("gain")), key=lambda x: x[1],
                                    reverse=True))[:200]
                            ))
            pre = model.predict(te_x, num_iteration=model.best_iteration)
            pred = model.predict(test_x, num_iteration=model.best_iteration)
            train[test_index] = pre
            test_pre[i, :] = pred
            cv_scores.append(log_loss(te_y, pre))
            
            f1_list=f1_score(te_y,np.argmax(pre,axis=1),average=None)
            f1=0.2*f1_list[0]+0.2*f1_list[1]+0.6*f1_list[2]
            
            f1_scores.append(f1)
            cv_rounds.append(model.best_iteration)
            # plt.figure(figsize=(12,6))
            # model.plot_importance(model, max_num_features=30)
            # plt.title("Featurertances")
            # plt.show()
            test_pre_all[i, :] = np.argmax(pred, axis=1)

        print("lgb now score is:" , cv_scores)
        print("lgb now f1-score is:", f1_scores)
        print("lgb now round is:", cv_rounds)
    test[:] = test_pre.mean(axis=0)
    print("lgb_score_list:", cv_scores)
    print("lgb_score_mean:", np.mean(cv_scores))
    print("lgb_score_std:" , np.std(cv_scores))
    print("f1_scores_mean:", np.mean(f1_scores))
    return train, test, test_pre_all, np.mean(f1_scores)


def lgb(x_train, y_train, x_valid):
    lgb_train, valid_y_pred, sb, cv_scores = train_lgb(x_train, y_train, x_valid, 3)
    return lgb_train, valid_y_pred, sb, cv_scores


def get_df_from_json(train_json, df_pkl_path):
    df_time = get_time_features(train_json[:],"amap_traffic_train_0712")
    import pandas as pd
    df_detect = postprocess_detect(df_pkl_path)
    # cos 相似度
    # df_cos = pd.read_pickle("/workdir/congest/result/cos_anno.pkl")
    # df_cos['cos_dif'] = df_cos['cos_max'] - df_cos['cos_min']
    # df_cos['cos_v_dif'] = df_cos['cos_v_max'] - df_cos['cos_v_min']
    # df_cos.rename(columns={'id': 'map_id'}, inplace=True)
    # df_time = pd.merge(df_time, df_cos, on=['map_id'])

    df_detect.rename(columns={'id': 'map_id'}, inplace=True)
    df_all = pd.merge(df_time, df_detect, on=['map_id'])
    return df_all

if __name__ == "__main__":
    result_path="../prediction_result"   #存放数据的地址
    mode = "submit" # local: 本地测试 submit: 提交
    # mode = "local" #: 本地测试 submit: 提交
    # select_features=[
    #                 "dayofweek_mean",
    #                 "person_nonvehicle_num_max",
    #                 "vehicle_num_key",
    #                 "max_area_min",
    #                 "max_area_key"
    #                 ]
    select_features=[
                    # "gap_mean",
                    # "gap_std",
                    # "hour_mean",
                    # "minute_mean",
                    "dayofweek_mean",
                    # "gap_time_today_mean",
                    # "gap_time_today_std",
                    # sim
                    # "cos_max",
                    # "cos_min",
                    # "cos_dif",
                    # "cos_mean",
                    # "cos_std",
                    # "cos_v_max",
                    # "cos_v_min",
                    # "cos_v_dif",
                    # "cos_v_mean",
                    # "cos_v_std",
                    # detect
                    "person_nonvehicle_num_max",
                    # "person_nonvehicle_num_min",
                    # "person_nonvehicle_num_cal_dif",
                    # "person_nonvehicle_num_key",
                    # "person_nonvehicle_num_mean",
                    # "person_nonvehicle_num_std",
                    # "vehicle_num_max",
                    # "vehicle_num_min",
                    # "vehicle_num_cal_dif",
                    "vehicle_num_key",
                    # "vehicle_num_mean",
                    # "vehicle_num_std",
                    # "max_area_max",
                    "max_area_min",
                    # "max_area_cal_dif",
                    "max_area_key",
                    # "max_area_mean",
                    # "max_area_std"
                    ]
    
    if mode == "submit":
        # 全部train数据用于训练
        train_json = pd.read_json("../data/amap_traffic_annotations_train.json")
        df_pkl_path = "../user_data/tmp_data/detection_train.pkl"
        train_df = get_df_from_json(train_json, df_pkl_path)
        #
        train_x=train_df[select_features].copy()
        train_y=train_df["label"]
        ##
        test_x = postprocess_detect_test()
        test_x.rename(columns={'id': 'map_id'}, inplace=True)
        test_json = pd.read_json("../data/amap_traffic_annotations_b_test_0828.json")
        df_pkl_path = "../user_data/tmp_data/detection_testb.pkl"
        test_df = get_df_from_json(test_json, df_pkl_path)
        test_x = test_df[select_features].copy()

        ##### lgb train #####
        lgb_train, lgb_test, sb, cv_scores = train_lgb(train_x, train_y, test_x, 3)
        sub=test_df[["map_id"]].copy()
        sub["pred"]=np.argmax(lgb_test,axis=1)

        result_dic=dict(zip(sub["map_id"],sub["pred"]))
        #保存
        import json
        with open("../data/amap_traffic_annotations_test.json", "r") as f:
            content=f.read()
        content=json.loads(content)
        for i in content["annotations"]:
            i['status']=result_dic[i["id"]]
        import os
        with open(os.path.join(result_path, "sub_%.4f.json"%cv_scores),"w") as f:
            f.write(json.dumps(content))
    else:
        # 用训练集做cross validation
        train_json = pd.read_json("../data/amap_traffic_annotations_train.json")
        # df_pkl_path = "../user_data/tmp_data/detection_train.pkl"
        
        ## DEBUG
        df_pkl_path = "../user_data/tmp_data/detection_train.pkl"
        ## DEBUG
        
        train_df = get_df_from_json(train_json, df_pkl_path)
        dfData = train_df[select_features].corr()
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.subplots(figsize=(20, 20)) # 设置画面大小
        sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
        plt.savefig('./BluesStateRelation.png')
        #
        train_x=train_df[train_df["train_valid"]==0][select_features].copy()
        train_y=train_df[train_df["train_valid"]==0]["label"]

        valid_x=train_df[train_df["train_valid"]==1][select_features].copy()
        valid_y_real=train_df[train_df["train_valid"]==1][["map_id","label"]]

        ##### lgb train #####
        lgb_train, np_valid_y_pred, sb, cv_scores = train_lgb(train_x, train_y, valid_x, 3)

        valid_y_pred=train_df[train_df["train_valid"]==1][["map_id"]].copy()
        valid_y_pred["pred"]=np.argmax(np_valid_y_pred,axis=1)

        # 本地计算
        valid_y_pred = pd.merge(valid_y_pred, valid_y_real, on=['map_id'])
        from sklearn.metrics import classification_report, confusion_matrix
        print(confusion_matrix(valid_y_pred['label'], valid_y_pred['pred']))
        print(classification_report(valid_y_pred['label'], valid_y_pred['pred']))

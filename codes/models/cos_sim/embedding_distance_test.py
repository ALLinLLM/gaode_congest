import torchvision.transforms
import torchvision.models as models
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import json
from PIL import Image
import numpy as np
from tqdm import tqdm  

class Vgg19Embedding(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        feature_layer= 35  # 35 last conv2d 38: after maxpool  42: 
        model = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(model.features.children())[:feature_layer])
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        # Assume input range is [0, 1]
        x = (x - self.mean) / self.std
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return x


class ConjestSingleImageDataset(data.Dataset):
    '''
    读取json, 只取key image
    
    图像序列的参考帧图像名
    图像序列的路况状态
    0：畅通，1：缓行，2：拥堵，-1：测试集真值未给出
    每帧图像采集时刻的GPS时间
    单位为秒。如GPS时间 1552806926 比 1552806921 滞后5秒钟

    '''
    def __init__(self, X_list, y_list, transforms):
        super(ConjestSingleImageDataset, self).__init__()
        self.X_list, self.y_list = X_list, y_list
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.X_list[index]
        target = self.y_list[index]
        img = Image.open(img_path)
        return self.transforms(img), target

    def __len__(self):
        return len(self.X_list)


class ConjestTestDataset(data.Dataset):
    '''
    读取json, 只取key image
    
    图像序列的参考帧图像名
    图像序列的路况状态
    0：畅通，1：缓行，2：拥堵，-1：测试集真值未给出
    每帧图像采集时刻的GPS时间
    单位为秒。如GPS时间 1552806926 比 1552806921 滞后5秒钟

    '''
    def __init__(self, X_list, y_list, transforms):
        super(ConjestTestDataset, self).__init__()
        self.X_list, self.y_list = X_list, y_list
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.X_list[index]
        target = self.y_list[index]
        img = Image.open(img_path)
        return self.transforms(img), img_path

    def __len__(self):
        return len(self.X_list)


def tSNE():
    from sklearn.manifold import TSNE 
    from pandas.core.frame import DataFrame
    import pandas as pd  
    import numpy as np  
    l=[]
    with open('1.csv','r') as fd:
    
        line= fd.readline()
        while line:
            if line =="":
                continue
    
            line = line.strip()
            word = line.split(",")
            l.append(word)
            line= fd.readline()
    
    data_l=DataFrame(l)
    print ("data_l ok")
    dataMat = np.array(data_l)  
    
    
    pca_tsne = TSNE(n_components=2)  
    newMat = pca_tsne.fit_transform(dataMat)  
    
    
    data1 = DataFrame(newMat)
    data1.to_csv('2.csv',index=False,header=False)


def main():
    expiremtent_name = "baseline"
    model_save_path = "../model_weights/baseline.pth"

    # init dataset
    with open("../data/all_train.txt", "r", encoding="utf8") as f:
        train_lines = f.read().splitlines()
    
    with open("../data/test.txt", "r", encoding="utf8") as f:
        test_lines = f.read().splitlines()

    X_list, y_list = [], []
    for line in train_lines:
        x, y = line.split(',')
        X_list.append(x)
        y_list.append(int(y))
    
    test_X_list, test_y_list = [], []
    for line in test_lines:
        x, y = line.split(',')
        test_X_list.append(x)
        test_y_list.append(int(y))

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
    print("train/test datasets length:", len(X_list), len(test_X_list))
    train_dataset = ConjestSingleImageDataset(X_list, y_list, train_transforms)
    test_dataset = ConjestTestDataset(test_X_list, test_y_list, train_transforms)
    batch_size = 164
    num_workers = 0
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers, drop_last=True, pin_memory=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False,
                                           num_workers=num_workers, drop_last=True, pin_memory=False)
    # baseline    
    model = Vgg19Embedding()
    model.cuda()
    summary(model, (3, 224, 224))

    # train
    is_first = 0
    with torch.no_grad():
        for _, (X, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            X = X.cuda()
            y.numpy()
            # 
            embed = model(X)
            embed = embed.cpu().numpy()
            if is_first == 0:
                a = np.hstack((y.unsqueeze(0).T.numpy(), embed))
                is_first = 1
            else:
                a = np.vstack((a, np.hstack((y.unsqueeze(0).T.numpy(), embed))))
                
    mean_embedings = {}
    for i in np.unique(a[:, 0]):
        tmp = a[np.where(a[:,0] == i)][:, 1:]
        mean_embedings[i] = torch.from_numpy(np.mean(tmp, 0)).cuda().unsqueeze(0)
    import os
    is_first = 0
    result = {}
    with torch.no_grad():
        for _, (X, y) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            X = X.cuda()
            # 
            embed = model(X)
            
            a_dist = 1-torch.cosine_similarity(embed, mean_embedings[0.0], dim=1)
            b_dist = 1-torch.cosine_similarity(embed, mean_embedings[1.0], dim=1)
            c_dist = 1-torch.cosine_similarity(embed, mean_embedings[2.0], dim=1)
            all_ = torch.stack((a_dist, b_dist, c_dist), dim=1)
            y_hat = torch.argmin(all_, dim=1)  # 计算每一行的最大值, 不要列了
            y_hat = y_hat.cpu().numpy()
            ii = 0 
            for ii in range(len(y)):
                _id = os.path.basename(os.path.dirname(y[ii]))
                status = y_hat[ii]
                result[_id] = status
    json_path = "/workdir/datasets/gaode_congest/amap_traffic_annotations_test.json"
    out_path = "/workdir/datasets/gaode_congest/amap_traffic_annotations_test_result.json"
    with open(json_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as w:
        json_dict = json.load(f)
        data_arr = json_dict["annotations"]  
        new_data_arr = [] 
        for data in data_arr:
            id_ = data["id"]
            data["status"] = int(result[id_])
            new_data_arr.append(data)
        json_dict["annotations"] = new_data_arr
        json.dump(json_dict, w)


if __name__ == "__main__":
    main()
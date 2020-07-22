import torchvision.transforms

from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import json
from PIL import Image
import numpy as np
from tqdm import tqdm  
from featureExtract import Vgg19Embedding
from sklearn.manifold import TSNE 

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



def main():
    torch.manual_seed(2020)
    expiremtent_name = "baseline"
    model_save_path = "../model_weights/baseline.pth"

    # init dataset
    with open("../../data/train_seed2020.txt", "r", encoding="utf8") as f:
        train_lines = f.read().splitlines()
    
    with open("../../data/valid_seed2020.txt", "r", encoding="utf8") as f:
        valid_lines = f.read().splitlines()

    X_list, y_list = [], []
    for line in train_lines:
        x, y = line.split(',')
        X_list.append(x)
        y_list.append(int(y))
    
    valid_X_list, valid_y_list = [], []
    for line in valid_lines:
        x, y = line.split(',')
        valid_X_list.append(x)
        valid_y_list.append(int(y))

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
    print("train/valid datasets length:", len(X_list), len(valid_X_list))
    train_dataset = ConjestSingleImageDataset(X_list, y_list, train_transforms)
    valid_dataset = ConjestSingleImageDataset(valid_X_list, valid_y_list, train_transforms)
    batch_size = 16
    num_workers = 0
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers, drop_last=True, pin_memory=False)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                           num_workers=num_workers, drop_last=True, pin_memory=False)
    # baseline    
    model = Vgg19Embedding()
    model.cuda()
    # summary(model, (3, 224, 224))

    # train
    is_first = 0
    with torch.no_grad():
        for ii, (X, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            if ii > 10:
                break
            X = X.cuda()
            y.numpy()
            # 
            embed = model(X)
            # embed = embed.cpu().numpy()
            if is_first == 0:
                X_train = embed
                y_train = y.unsqueeze(0).T  # batch_size, 1
                is_first = 1
            else:
                X_train = torch.cat((X_train, embed), dim=0)
                y_train = torch.cat((y_train, y.unsqueeze(0).T), dim=0)
                # X_train = np.vstack((X_train, embed))
                # y_train = np.vstack((y_train, y.unsqueeze(0).T.numpy()))
    
    print("train X_train", X_train.shape)
    
    is_first = 0
    with torch.no_grad():
        for ii, (X, y) in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
            if ii > 10:
                break
            X = X.cuda()
            # 
            embed = model(X)

            if is_first==0:
                X_valid = embed
                y_valid_real = y.unsqueeze(0).T
                is_first=1
            else:
                X_valid = torch.cat((X_valid, embed), dim=0)
                y_valid_real = torch.cat((y_valid_real, y.unsqueeze(0).T), dim=0)


    y_valid_zeros = torch.zeros(y_valid_real.shape).float() - 1
    
    # 合并train和valid的向量, 准备降维
    X_all = torch.cat((X_train, X_valid), dim=0)
    y_all = torch.cat((y_train.float(), y_valid_zeros), dim=0)
    
    # 复制到cpu
    X_all = X_all.cpu().detach().numpy()
    y_all = y_all.cpu().detach().numpy()

    # 降维
    tsne = TSNE(n_components=2)  
    X_all_2d = tsne.fit_transform(X_all)  
    a = np.hstack((y_all, X_all_2d))
    print(a.shape)
    mean_embedings = {}
    for i in np.unique(a[:, 0]):
        tmp = a[np.where(a[:,0] == i)][:, 1:]
        if i == -1:
            X_valid_2d = tmp 
        else:
            mean_embedings[i] = torch.from_numpy(np.mean(tmp, 0)).cuda().unsqueeze(0)

    
    # t-SNE降维对新的数据怎么办
    X_valid_2d = torch.from_numpy(X_valid_2d).cuda()
    a_dist = 1-torch.cosine_similarity(X_valid_2d, mean_embedings[0.0], dim=1)
    b_dist = 1-torch.cosine_similarity(X_valid_2d, mean_embedings[1.0], dim=1)
    c_dist = 1-torch.cosine_similarity(X_valid_2d, mean_embedings[2.0], dim=1)
    all_ = torch.stack((a_dist, b_dist, c_dist), dim=1)
    y_valid_predict = torch.argmin(all_, dim=1)  # 计算每一行的最大值, 不要列了
    y_valid_predict = y_valid_predict.squeeze().cpu()
    # from sklearn.metrics import accuracy_score
    # from sklearn.metrics import precision_score
    # from sklearn.metrics import recall_score
    # from sklearn.metrics import f1_score
    # print("++++++++ accuracy")
    # print(accuracy_score(y, y_valid_predict))  # 0.5
    # print("++++++++ macro")
    # print(precision_score(y, y_valid_predict, average='macro'))
    # print(recall_score(y, y_valid_predict, average='macro'))  # 0.3333333333333333
    # print(f1_score(y, y_valid_predict, average='macro'))  # 0.26666666666666666
    
    # print("++++++++ micro")
    # print("precision", precision_score(y, y_valid_predict, average='micro'))
    # print("recall", recall_score(y, y_valid_predict, average='micro'))  # 0.3333333333333333
    # print("f1", f1_score(y, y_valid_predict, average='micro'))  # 0.3333333333333333

    from sklearn.metrics import classification_report
    print(classification_report(y_valid_real, y_valid_predict))

if __name__ == "__main__":
    main()
import torchvision.transforms
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
from data import get_trainXy_validXy

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
        img0 = Image.open(img_path[0])
        img1 = Image.open(img_path[1])
        return self.transforms(img0), self.transforms(img1), target

    def __len__(self):
        return len(self.X_list)



def main():
    print("start result")
    feature_layers = 37
    torch.manual_seed(2020)
    use_tSNE = False
    expiremtent_name = "baseline"
    model_save_path = "../model_weights/baseline.pth"

    # -1: 无限
    total_num = -1 
    valid_num = 200
    # init dataset
    X_list, y_list, valid_X_list, valid_y_list = get_trainXy_validXy(total_num, valid_num)
    print("train/valid datasets length:", len(X_list), len(valid_X_list))
    # 1300 200

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
    train_dataset = ConjestSingleImageDataset(X_list, y_list, train_transforms)
    valid_dataset = ConjestSingleImageDataset(valid_X_list, valid_y_list, train_transforms)
    batch_size = 16
    num_workers = 4
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers, drop_last=True, pin_memory=False)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                           num_workers=num_workers, drop_last=True, pin_memory=False)
    # baseline    
    model = Vgg19Embedding(feature_layers)
    model = model.cuda()
    print(model)
    # from torchsummary import summary
    # summary(model, (3, 224, 224))

    # train
    is_first = 0
    with torch.no_grad():
        for ii, (X0, X1, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # if ii > 10:
            #     break
            X0 = X0.cuda()  # require_grads == False
            X1 = X1.cuda()  # require_grads == False
            # 
            embed0 = model(X0) # 25088
            embed1 = model(X1)
            cos_sim = torch.cosine_similarity(embed1, embed0, dim=1).unsqueeze(1)
            # embed = embed.cpu().numpy()
            if is_first == 0:
                X_train = cos_sim
                y_train = y.unsqueeze(0).T  # batch_size, 1
                is_first = 1
            else:
                X_train = torch.cat((X_train, cos_sim), dim=0)
                y_train = torch.cat((y_train, y.unsqueeze(0).T), dim=0)
                # X_train = np.vstack((X_train, embed))
                # y_train = np.vstack((y_train, y.unsqueeze(0).T.numpy()))
    # y_train[0]=2
    print("valid..")
    # if valid_num>-1:
    is_first = 0
    with torch.no_grad():
        for ii, (X0, X1, y) in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
            # if ii > 10:
            #     break
            X0 = X0.cuda()  # require_grads == False
            X1 = X1.cuda()  # require_grads == False
            # 
            embed0 = model(X0) # 25088
            embed1 = model(X1)
            cos_sim = torch.cosine_similarity(embed1, embed0, dim=1).unsqueeze(1)
            if is_first==0:
                X_valid = cos_sim
                y_valid_real = y.unsqueeze(0).T
                is_first=1
            else:
                X_valid = torch.cat((X_valid, cos_sim), dim=0)
                y_valid_real = torch.cat((y_valid_real, y.unsqueeze(0).T), dim=0)


    y_valid_zeros = torch.zeros(y_valid_real.shape).float() - 1
    
    # 合并train和valid的向量, 准备降维
    X_all = torch.cat((X_train, X_valid), dim=0)
    y_all = torch.cat((y_train.float(), y_valid_zeros), dim=0)
    
    # 复制到cpu
    X_all = X_all.cpu().detach().numpy()
    y_all = y_all.cpu().detach().numpy()

    if use_tSNE:
        # 降维
        tsne = TSNE(n_components=2)  
        X_all_2d = tsne.fit_transform(X_all)  
        a = np.hstack((y_all, X_all_2d))
    else:
        a = np.hstack((y_all, X_all))

    print("embeding shape:", a.shape)
    mean_embedings = []
    for i in np.unique(a[:, 0]):
        tmp = a[np.where(a[:,0] == i)][:, 1:]
        if i == -1:
            X_valid_2d = tmp 
        else:
            mean_embedings.append(np.mean(tmp))
    X_valid_2d
    y_valid_predict = np.argmin(np.abs(np.array(mean_embedings) - X_valid_2d), 1)

    from sklearn.metrics import classification_report
    with open("result.txt", "w") as f:
        f.write(classification_report(y_valid_real.numpy(), y_valid_predict))

if __name__ == "__main__":
    main()
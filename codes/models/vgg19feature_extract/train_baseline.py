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

from torch.utils.tensorboard import SummaryWriter


class ConjestNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model = models.vgg19(pretrained=True)
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.fc = nn.Linear(1000, 3)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        # Assume input range is [0, 1]
        x = (x - self.mean) / self.std
        x = self.model(x)
        x = self.fc(x)
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


def train():
    expiremtent_name = "baseline"
    model_save_path = "../model_weights/baseline.pth"

    # init dataset
    with open("../data/train_seed2020.txt", "r", encoding="utf8") as f:
        train_lines = f.read().splitlines()
    
    with open("../data/test_seed2020.txt", "r", encoding="utf8") as f:
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
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ])
    print("train/test datasets length:", len(X_list), len(test_X_list))
    train_dataset = ConjestSingleImageDataset(X_list, y_list, train_transforms)
    test_dataset = ConjestSingleImageDataset(test_X_list, test_y_list, train_transforms)
    batch_size = 32
    num_workers = 0
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers, drop_last=True, pin_memory=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False,
                                           num_workers=num_workers, drop_last=True, pin_memory=False)
    # baseline    
    model = ConjestNet()
    model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(
                    model.parameters(), lr=1e-4,
                    weight_decay=0)

    # 
    train_tb_logger = SummaryWriter(log_dir='../tb_logger/congest/'+expiremtent_name+"/train")
    test_tb_logger = SummaryWriter(log_dir='../tb_logger/congest/'+expiremtent_name+"/valid")

    # train
    total_epochs = 10
    print("total_epochs: %d, batchsize: %d, each epoch has %d iters"%(total_epochs, batch_size, len(train_dataset)//batch_size))
    total_iters = total_epochs*(len(train_dataset)//batch_size)
    print("total_iters: %d"%(total_iters))

    test_iter = 50
    print_iter = 50
    save_iter = 10
    current_iter = 0
    for epoch in range(1, total_epochs+1):
        for _, (X, y) in enumerate(train_dataloader):
            current_iter += 1
            X = X.cuda()
            y = y.cuda()
            # 
            optimizer.zero_grad()
            y_hat = model(X)
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            # update
            optimizer.step()
            if current_iter%print_iter==0:
                loss_ = loss.detach().cpu().item()
                print("current_iters/total_iters: %d/%d, loss: %.4f"%(current_iter, total_iters, loss_))
                train_tb_logger.add_scalar('loss', loss_, current_iter)
            if current_iter%test_iter==0:
                print("++++++++++ valid")
                valid_mean_loss = 0
                num = 0
                for _, (p, q) in enumerate(test_dataloader):
                    num += 1
                    p = p.cuda()
                    q = q.cuda()
                    q_hat = model(p)
                    loss = F.cross_entropy(q_hat, q)
                    loss_ = loss.detach().cpu().item()
                    valid_mean_loss += loss_
                valid_mean_loss /= num
                test_tb_logger.add_scalar('loss', valid_mean_loss, current_iter)
        if epoch%save_iter==0:
            torch.save(model.state_dict(), model_save_path.replace(".pth", "_epoch%d.pth"%epoch))
    # test_json_path = "amap_traffic_annotations_test.json"
    # test_dataset = ConjestSingleImageDataset(json_path)
    # batch_size = 32
    # shuffle = False
    # num_workers = 6
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
    #                                        num_workers=num_workers, drop_last=True, pin_memory=False)
# summary(model.cuda(), (3, 224, 224))
# feature_layer=37
# features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
# summary(features.cuda(), (3, 224, 224))

# for k, v in self.features.named_parameters():
#     v.requires_grad = False

if __name__ == "__main__":
    train()

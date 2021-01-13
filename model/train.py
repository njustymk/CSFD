import os
import pdb
import math
import torchvision
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms as transforms

from dataset import irfaceDataset

# class module2(nn.Module):
#     def __init__(self):
#         super(module2, self).__init__()
#         self.features = torchvision.models.vgg16(pretrained=True).features
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 8),
#         )
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x.view(x.size(0), -1))
#         return x

class module2(nn.Module):
    def __init__(self):
        super(module2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 8),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss

def train(epoch, model, train_loader, optimizer, loss_func, device):
    print("train:")
    train_loss = 0
    for batch_num, (data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11) in enumerate(train_loader):

        output1 = model(data1.to(device))
        output2 = model(data2.to(device))
        output3 = model(data3.to(device))
        output4 = model(data4.to(device))
        output5 = model(data5.to(device))
        output6 = model(data6.to(device))
        output7 = model(data7.to(device))
        output8 = model(data8.to(device))
        output9 = model(data9.to(device))
        output10 = model(data10.to(device))
        output11 = model(data11.to(device))

        loss3 = loss_func(output1, output2, output3)
        loss4 = loss_func(output1, output2, output4)
        loss5 = loss_func(output1, output2, output5)
        loss6 = loss_func(output1, output2, output6)
        loss7 = loss_func(output1, output2, output7)
        loss8 = loss_func(output1, output2, output8)
        loss9 = loss_func(output1, output2, output9)
        loss10 = loss_func(output1, output2, output10)
        loss11 = loss_func(output1, output2, output11)

        # loss = loss3
        # loss = loss3+loss4+loss5+loss6+loss7
        loss = loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10+loss11

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        print(f'\r "loss: "{loss:.08f} train_loss:{train_loss/(batch_num+1):.08f}      ', end='')

def test(model, test_loader, loss_func, device):
    print("\ntest:")
    test_loss = 0
    for batch_num, (data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11) in enumerate(test_loader):

        output1 = model(data1.to(device))
        output2 = model(data2.to(device))
        output3 = model(data3.to(device))
        output4 = model(data4.to(device))
        output5 = model(data5.to(device))
        output6 = model(data6.to(device))
        output7 = model(data7.to(device))
        output8 = model(data8.to(device))
        output9 = model(data9.to(device))
        output10 = model(data10.to(device))
        output11 = model(data11.to(device))

        loss3 = loss_func(output1, output2, output3)
        loss4 = loss_func(output1, output2, output4)
        loss5 = loss_func(output1, output2, output5)
        loss6 = loss_func(output1, output2, output6)
        loss7 = loss_func(output1, output2, output7)
        loss8 = loss_func(output1, output2, output8)
        loss9 = loss_func(output1, output2, output9)
        loss10 = loss_func(output1, output2, output10)
        loss11 = loss_func(output1, output2, output11)

        # loss = loss3
        # loss = loss3+loss4+loss5+loss6+loss7
        loss = loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10+loss11

        test_loss+=loss.item()
        print(f'\r "loss: "{loss:.08f} test_loss:{test_loss/(batch_num+1):.08f}      ', end='')

def choice(model):
    # print('=====================================')
    # print('===             1.Adam            ===')
    # print('===             2.SGD             ===')
    # print('===             3.RMSprop         ===')
    # print('=====================================')
    # choices = int(input())
    # if choices == 1:
    #     # 选用Adam作为优化器，学习率为0.001
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)
    # elif choices == 2:
    #     optimizer = optim.SGD(model.parameters(), lr=0.001)
    # elif choices == 3:
    #     optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return optimizer


# 运行函数，进行模型训练和测试集的测试
def run(data_root, weights_dir, device):

    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_set = irfaceDataset(root=data_root, datasplit='train', transform=train_transform)  # 训练数据集
    test_set = irfaceDataset(root=data_root, datasplit='test', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=128, shuffle=False)

    cudnn.benchmark = True
    model = module2().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    loss_func = TripletLoss()
    epochs = 50

    for epoch in range(1, epochs + 1):
        print('\n===> epoch: {}/{}'.format(epoch, epochs))
        train(epoch - 1, model, train_loader, optimizer, loss_func, device)
        test(model, test_loader, loss_func, device)
        torch.save(model, weights_dir+'/vgg11_data_epoch{}.pt'.format(epoch))


if __name__ == '__main__':

    data_root = '../data/train/face'
    print('data_root', data_root)

    weights_dir = 'weights_9neg' #weights_1neg, weights_5neg, weights_9neg
    if not os.path.exists(weights_dir): os.mkdir(weights_dir)

    if torch.cuda.is_available(): device = 'cuda'
    else: device = 'cpu'
    
    run(data_root, weights_dir, device)


from __future__ import print_function, division

import json

import torch.nn.functional as F
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.model_zoo as model_zoo
from torchvision import datasets, transforms
from torchvision import models
from torch.autograd import Variable
#import torchsummary as summary
import os
import csv
import codecs
import numpy as np
import time
#from thop import profile

# from ft.OCI_Iterator import OCIIterator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
'''定义超参数'''
EPOCH = 20
batch_size = 32
classes_num = 1000
learning_rate = 1e-3

DEVICE = torch.device("cuda:1")

'''定义Transform'''
# 对训练集做一个变换
train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),  # 对图片尺寸做一个缩放切割
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.ToTensor(),  # 转化为张量
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 进行归一化
])
# 对测试集做变换
val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dir = r"/root/mini-imagenet/train"  # 训练集路径
# 定义数据集
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
# 加载数据集
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=8,
                                               pin_memory=True)  # ,num_workers=16,pin_memory=False

val_dir = "/root/mini-imagenet/val"
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=8,
                                             pin_memory=True)  # ,num_workers=16,pin_memory=True

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Reduction_A(nn.Module):
    # 35 -> 17
    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch_0 = Conv2d(in_channels, n, 3, stride=2, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, k, 1, stride=1, padding=0, bias=False),
            Conv2d(k, l, 3, stride=1, padding=1, bias=False),
            Conv2d(l, m, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1) # 17 x 17 x 1024

class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.conv2d_1a_3x3 = Conv2d(in_channels, 32, 3, stride=2, padding=0, bias=False)

        self.conv2d_2a_3x3 = Conv2d(32, 32, 3, stride=1, padding=0, bias=False)
        self.conv2d_2b_3x3 = Conv2d(32, 64, 3, stride=1, padding=1, bias=False)

        self.mixed_3a_branch_0 = nn.MaxPool2d(3, stride=2, padding=0)
        self.mixed_3a_branch_1 = Conv2d(64, 96, 3, stride=2, padding=0, bias=False)

        self.mixed_4a_branch_0 = nn.Sequential(
            Conv2d(160, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 96, 3, stride=1, padding=0, bias=False),
        )
        self.mixed_4a_branch_1 = nn.Sequential(
            Conv2d(160, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 64, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(64, 64, (7, 1), stride=1, padding=(3, 0), bias=False),
            Conv2d(64, 96, 3, stride=1, padding=0, bias=False)
        )

        self.mixed_5a_branch_0 = Conv2d(192, 192, 3, stride=2, padding=0, bias=False)
        self.mixed_5a_branch_1 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x = self.conv2d_1a_3x3(x) # 149 x 149 x 32
        x = self.conv2d_2a_3x3(x) # 147 x 147 x 32
        x = self.conv2d_2b_3x3(x) # 147 x 147 x 64
        x0 = self.mixed_3a_branch_0(x)
        x1 = self.mixed_3a_branch_1(x)
        x = torch.cat((x0, x1), dim=1) # 73 x 73 x 160
        x0 = self.mixed_4a_branch_0(x)
        x1 = self.mixed_4a_branch_1(x)
        x = torch.cat((x0, x1), dim=1) # 71 x 71 x 192
        x0 = self.mixed_5a_branch_0(x)
        x1 = self.mixed_5a_branch_1(x)
        x = torch.cat((x0, x1), dim=1) # 35 x 35 x 384
        return x


class Inception_A(nn.Module):
    def __init__(self, in_channels):
        super(Inception_A, self).__init__()
        self.branch_0 = Conv2d(in_channels, 96, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 96, 3, stride=1, padding=1, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 96, 3, stride=1, padding=1, bias=False),
            Conv2d(96, 96, 3, stride=1, padding=1, bias=False),
        )
        self.brance_3 = nn.Sequential(
            nn.AvgPool2d(3, 1, padding=1, count_include_pad=False),
            Conv2d(384, 96, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.brance_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inception_B(nn.Module):
    def __init__(self, in_channels):
        super(Inception_B, self).__init__()
        self.branch_0 = Conv2d(in_channels, 384, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv2d(192, 224, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(224, 256, (7, 1), stride=1, padding=(3, 0), bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv2d(192, 192, (7, 1), stride=1, padding=(3, 0), bias=False),
            Conv2d(192, 224, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(224, 224, (7, 1), stride=1, padding=(3, 0), bias=False),
            Conv2d(224, 256, (1, 7), stride=1, padding=(0, 3), bias=False)
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(in_channels, 128, 1, stride=1, padding=0, bias=False)
        )
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Reduction_B(nn.Module):
    # 17 -> 8
    def __init__(self, in_channels):
        super(Reduction_B, self).__init__()
        self.branch_0 = nn.Sequential(
            Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv2d(192, 192, 3, stride=2, padding=0, bias=False),
        )
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 256, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(256, 320, (7, 1), stride=1, padding=(3, 0), bias=False),
            Conv2d(320, 320, 3, stride=2, padding=0, bias=False)
        )
        self.branch_2 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1)  # 8 x 8 x 1536


class Inception_C(nn.Module):
    def __init__(self, in_channels):
        super(Inception_C, self).__init__()
        self.branch_0 = Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False)

        self.branch_1 = Conv2d(in_channels, 384, 1, stride=1, padding=0, bias=False)
        self.branch_1_1 = Conv2d(384, 256, (1, 3), stride=1, padding=(0, 1), bias=False)
        self.branch_1_2 = Conv2d(384, 256, (3, 1), stride=1, padding=(1, 0), bias=False)

        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 384, 1, stride=1, padding=0, bias=False),
            Conv2d(384, 448, (3, 1), stride=1, padding=(1, 0), bias=False),
            Conv2d(448, 512, (1, 3), stride=1, padding=(0, 1), bias=False),
        )
        self.branch_2_1 = Conv2d(512, 256, (1, 3), stride=1, padding=(0, 1), bias=False)
        self.branch_2_2 = Conv2d(512, 256, (3, 1), stride=1, padding=(1, 0), bias=False)

        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x1_1 = self.branch_1_1(x1)
        x1_2 = self.branch_1_2(x1)
        x1 = torch.cat((x1_1, x1_2), 1)
        x2 = self.branch_2(x)
        x2_1 = self.branch_2_1(x2)
        x2_2 = self.branch_2_2(x2)
        x2 = torch.cat((x2_1, x2_2), dim=1)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1) # 8 x 8 x 1536


class Inceptionv4(nn.Module):
    def __init__(self, in_channels=3, classes=1000, k=192, l=224, m=256, n=384):
        super(Inceptionv4, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for i in range(4):
            blocks.append(Inception_A(384))
        blocks.append(Reduction_A(384, k, l, m, n))
        for i in range(7):
            blocks.append(Inception_B(1024))
        blocks.append(Reduction_B(1024))
        for i in range(3):
            blocks.append(Inception_C(1536))
        self.features = nn.Sequential(*blocks)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

# --------------------训练过程---------------------------------
model = Inceptionv4()  # 在这里更换你需要训练的模型
model_name = r'inceptionv4'
training_result_dir = r'./training_result'
model = model.to(DEVICE)
# summary.summary(model, input_size=(3, 224, 224), device='cuda')  # 我们选择图形的出入尺寸为(3,224,224)

# params = [{'params': md.parameters()} for md in model.children()
#           if md in [model.classifier]]
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 按训练批次调整学习率，每30个epoch调整一次
# loss_func = nn.CrossEntropyLoss()
loss_func = F.cross_entropy
# 存储测试loss和acc
# Loss_list = []
# Accuracy_list = []
# 存储训练loss和acc
# train_Loss_list = []
# train_Accuracy_list = []
# 这俩作用是为了提前开辟一个
# loss = []
# loss1 = []
train_time_list = []
train_loss_list = []

# OCI = OCIIterator(model_name='resnet101',
#                   dataloader=train_dataloader,
#                   ft_lambda=0.0042,  # 4 hours a error
#                   ck_mode='MANUAL',
#                   ts=0.0015,  # the unit is min
#                   theta_1=-0.078,
#                   theta_2=0.787,
#                   epoch=EPOCH,
#                   ft_strategy='CCM',
#                   fit_interval=10000,
#                   profile_threshold=50,
#                   model=model,
#                   optimizer=optimizer)

def train_res(model, train_dataloader, since_t, epoch, train_time_list, train_loss_list):
    model.train()
    # print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    print(f'len(train_dataloader):{len(train_dataloader)} and len(train_datasets):{len(train_datasets)}')
    print()
    for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
        # iter_since = time.time()
        batch_x = Variable(batch_x).to(DEVICE)
        batch_y = Variable(batch_y).to(DEVICE)
        optimizer.zero_grad()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        # print(f'loss is {loss.item()}')
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        loss.backward()
        optimizer.step()
        # OCI.optimizer_step(loss, model, optimizer)
        # iter_end = time.time()
        # print(f'{batch_idx + 1:05d} iter takes {iter_end - iter_since:.4f}s')

        current_t = (time.time() - since_t) / 60
        train_time_list.append(current_t)
        train_loss_list.append(loss.item())
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch + 1:03d}/{EPOCH:03d} | '
                  f'Batch {batch_idx:04d}/{len(train_dataloader):04d} | '
                  f'Cost: {loss:.4f} | '
                  f'Time: {current_t:.4f}min')

        if batch_idx != 0 and batch_idx % 1251 == 0:
            print(f'batch_idx iter takes {(time.time() - since_t) / 60}min')

    return train_time_list, train_loss_list
    # print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_datasets)),
    #                                                train_acc / (len(train_datasets))))  # 输出训练时的loss和acc
    # train_Loss_list.append(train_loss / (len(val_datasets)))
    # train_Accuracy_list.append(100 * train_acc / (len(val_datasets)))


# evaluation--------------------------------
def val(model, val_dataloader):
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_idx, (batch_x, batch_y) in enumerate(val_dataloader):
        batch_x = Variable(batch_x, volatile=True).to(DEVICE)
        batch_y = Variable(batch_y, volatile=True).to(DEVICE)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(val_datasets)),
                                                  eval_acc / (len(val_datasets))))  # 输出测试时的loss和acc
    # Loss_list.append(eval_loss / (len(val_datasets)))
    # Accuracy_list.append(100 * eval_acc / (len(val_datasets)))

log_dir = './inceptionv4.pth'
def main():
    model.to(DEVICE)

    test_flag = False
    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    if os.path.exists(log_dir) and test_flag:
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        pre_train_t = checkpoint['pre_train_t']
        train_time_list = checkpoint['train_time_list']
        train_loss_list = checkpoint['train_loss_list']
        start_time = time.time() - pre_train_t
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        start_time = time.time()
        train_time_list = []
        train_loss_list = []
        print('无保存模型，将从头开始训练！')


    # start_epoch = 0
    for epoch in range(start_epoch, EPOCH):
        since = time.time()
        # print('epoch {}'.format(epoch))  # 显示每次训练次数
        train_time_list, train_loss_list = train_res(model, train_dataloader, start_time, epoch, train_time_list, train_loss_list)
        time_elapsed = time.time() - since
        print('An epoch takes in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 输出训练和测试的时间

        js = json.dumps(train_time_list)  # the unit is min
        f = open(os.path.join(training_result_dir, model_name + '_time.txt'), 'w+')
        f.write(js)
        f.close()

        js = json.dumps(train_loss_list)
        f = open(os.path.join(training_result_dir, model_name + '_loss.txt'), 'w+')
        f.write(js)
        f.close()

        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch,
                 'pre_train_t': time.time() - start_time,
                 'train_time_list': train_time_list,
                 'train_loss_list': train_loss_list}
        torch.save(state, log_dir)
        # 通过一个if语句判断，让模型每十次评估一次模型并且保存一次模型参数
        # epoch_num = epoch / 10
        # epoch_numcl = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 18.0,
        #                19.0, 20.0, 21.0]
        # print('epoch_num', epoch_num)
        # if epoch_num in epoch_numcl:
        #     print('评估模型')
        #     val(model, val_dataloader)
        #     print('保存模型')
        #     state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        #     torch.save(state, log_dir)

    # y1 = Accuracy_list
    # y2 = Loss_list
    # y3 = train_Accuracy_list
    # y4 = train_Loss_list
    #
    # x1 = range(len(Accuracy_list))
    # x2 = range(len(Loss_list))
    # x3 = range(len(train_Accuracy_list))
    # x4 = range(len(train_Loss_list))
    #
    # plt.subplot(2, 1, 1)
    # plt.plot(x1, y1, '-')
    # plt.title('Test accuracy vs. epoches')
    # plt.ylabel('Test accuracy')
    # plt.subplot(2, 1, 2)
    # plt.plot(x2, y2, '-')
    # plt.xlabel('Test loss vs. epoches')
    # plt.ylabel('Test loss')
    # plt.show()
    #
    # plt.subplot(2, 1, 1)
    # plt.plot(x3, y3, '-')
    # plt.title('Train accuracy vs. epoches')
    # plt.ylabel('Train accuracy')
    # plt.subplot(2, 1, 2)
    # plt.plot(x4, y4, '-')
    # plt.xlabel('Train loss vs. epoches')
    # plt.ylabel('Train loss')
    # plt.show()

    js = json.dumps(train_time_list)  # the unit is min
    f = open(os.path.join(training_result_dir, model_name + '_time.txt'), 'w+')
    f.write(js)
    f.close()

    js = json.dumps(train_loss_list)
    f = open(os.path.join(training_result_dir, model_name + '_loss.txt'), 'w+')
    f.write(js)
    f.close()


if __name__ == '__main__':
    main()

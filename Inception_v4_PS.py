from __future__ import print_function, division

import json

import torch.nn.functional as F
#import matplotlib.pyplot as plt
import torch
import copy
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
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from mpi4py import MPI
import argparse

def parse_devices(device_string):
    if device_string is None:
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in device_string.split(',')]

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--Flag', '-f', help='flag of first time to run', required=True)
parser.add_argument('--GPU_list', '-g', help='placement of each process', required=True)
parser.add_argument('--Id_index', '-i', help='index for job (checkpoint)', required=True)
parser.add_argument('--Epoch', '-e', help='epoch size:40 60 80', required=True)
args = parser.parse_args()

gpu_list = parse_devices(args.GPU_list)
first_run_flag = str(args.Flag)
job_id = str(args.Id_index) #The index of job to find the right checkpoint
model_dir = "/root/ExplSched/Scheduling/check_point/" + job_id + "_model.pt"
epoch_dir = "/root/ExplSched/Scheduling/check_point/" + job_id + "_epoch.pt"
#from thop import profile

# from ft.OCI_Iterator import OCIIterator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
'''定义超参数'''

batch_size = 64
classes_num = 1000
learning_rate = 1e-3

EPOCH=int(str(args.Epoch))
DEVICE = torch.device("cuda:1")
COMM = MPI.COMM_WORLD
rank=COMM.Get_rank()
size=COMM.Get_size()
# if gpu_list[rank] == 0:
#     os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#     device = torch.device('cuda:0')
# elif gpu_list[rank] == 1:
#     os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#     device = torch.device('cuda:0')
# elif gpu_list[rank] == 2:
#     os.environ["CUDA_VISIBLE_DEVICES"] = '2'
#     device = torch.device('cuda:0')
# else:
#     os.environ["CUDA_VISIBLE_DEVICES"] = '3'
#     device = torch.device('cuda:0')
DEVICE = torch.device("cuda:1")
device=DEVICE
if gpu_list[rank] == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda:0')
elif gpu_list[rank] == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device('cuda:0')
elif gpu_list[rank] == 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    device = torch.device('cuda:0')
elif gpu_list[rank] == 3:
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    device = torch.device('cuda:0')
elif gpu_list[rank] == 4:
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    device = torch.device('cuda:0')
elif gpu_list[rank] == 5:
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    device = torch.device('cuda:0')
elif gpu_list[rank] == 6:
    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    device = torch.device('cuda:0')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    device = torch.device('cuda:0')

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


train_dir = "/root/mini-imagenet/train"  # 训练集路径
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)

val_dir = "/root/mini-imagenet/val"
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)

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
model_name = 'inceptionv4'
training_result_dir = './training_result'

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

def distributed_samper(dataset, num_workers):
    dataset_list = []
    total_datasize = len(dataset)
    dataset_temp = copy.deepcopy(dataset)
    if total_datasize%num_workers!=0:
        dataset,abort=random_split(dataset=dataset,lengths=[int(total_datasize-total_datasize%num_workers),
                                                            total_datasize%num_workers])
        total_datasize = len(dataset)
        dataset_temp = copy.deepcopy(dataset)
    for i in range(num_workers):
        cuurent_size = len(dataset_temp)
        if cuurent_size == total_datasize / (num_workers):
            dataset_list.append(dataset_temp)
            break 
        dataset1, dataset2 = random_split(dataset=dataset_temp, lengths=[int(total_datasize / (num_workers)), int(cuurent_size - total_datasize / (num_workers))])
        dataset_list.append(dataset1)
        dataset_temp = dataset2
    return dataset_list

def grad_avg(g):
    #print(g[0])
    ret=copy.deepcopy(g[0])
    for i in range(len(ret)):
        for tt in range(1,len(g)):
            g[tt][i]=g[tt][i].cuda()
            ret[i]=ret[i].cuda()
            ret[i]+=g[tt][i]
        # ret[i] = torch.div(ret[i], len(g))
    return ret



def train_res(model, train_dataloader, epoch,t):
    start_time=time.time()
    model.train()
    # print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
        # iter_since = time.time()
        batch_x = Variable(batch_x).to(device)
        batch_y = Variable(batch_y).to(device)
        optimizer.zero_grad()
        out = model(batch_x).to(device)
        loss = loss_func(out, batch_y)
        # print(f'loss is {loss.item()}')
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        loss.backward()
        optimizer.step()

        time_dir = "/root/ExplSched/Scheduling/time_loss/" + "Inception_"+job_id+"_"+str(rank)+"_time.txt"
        with open(time_dir,"a") as time_t:
            
            if batch_idx % 100 == 0:
                current_t = (time.time() - t) / 60
                epoch_t=(time.time()-start_time)/60
                time_loss=str(epoch)+' '+str(loss.item())+' '+str(current_t)+' '+str(epoch_t)
                time_t.write(time_loss)
                time_t.write('\n')
        # OCI.optimizer_step(loss, model, optimizer)
        # iter_end = time.time()
        # print(f'{batch_idx + 1:05d} iter takes {iter_end - iter_since:.4f}s')

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
        batch_x = Variable(batch_x, volatile=True).to(device)
        batch_y = Variable(batch_y, volatile=True).to(device)
        out = model(batch_x).to(device)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
      # 输出测试时的loss和acc
    # Loss_list.append(eval_loss / (len(val_datasets)))
    # Accuracy_list.append(100 * eval_acc / (len(val_datasets)))


BUFFSIZE=2**32
Mpi_buf = bytearray(BUFFSIZE)

log_dir = './inceptionv4.pth'
if __name__ == '__main__':
    
    

    dataset_train_list = distributed_samper(dataset=train_datasets, num_workers=(size-2)*2) 
    dataset_test_list = distributed_samper(dataset=val_datasets, num_workers=(size-2)*2)
    local_batch_size = int(batch_size/((size-2)*2))

    # if first_run_flag == "true": #if the flag is True, represent the MPI process is not the first time to run 
    #         #print("this code run")
    #     

    if rank!=0 and rank!=1:
        model.to(device)
        start_epoch=1
        if first_run_flag == "true": #if the flag is True, represent the MPI process is not the first time to run 
            #print("this code run")
            model.load_state_dict(torch.load(model_dir))
            start_epoch = torch.load(epoch_dir)
            start_epoch = start_epoch + 1
            print(start_epoch) 
        data_loader = DataLoader(dataset_train_list[rank-1], local_batch_size, shuffle=True) 
        t=time.time()
        for epoch in range(start_epoch, EPOCH+1):
            train_res(model, data_loader, epoch,t)
            grad_list=[]
            for p in model.parameters():
                grad=p.grad
                grad_list.append(grad)
            MPI.Attach_buffer(Mpi_buf)
            COMM.bsend(grad_list,dest=1,tag=999)
            param_glob=COMM.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG)
            MPI.Detach_buffer()
            model.load_state_dict(param_glob)
            val_dataloader=DataLoader(dataset_test_list[rank-1], local_batch_size, shuffle=True) 
            val(model, val_dataloader)
            torch.save(epoch, epoch_dir,_use_new_zipfile_serialization=False)
            
    elif rank==1:
        model.to(device)
        if first_run_flag == "true": #if the flag is True, represent the MPI process is not the first time to run 
            #print("this code run")
                model.load_state_dict(torch.load(model_dir))

        while(1):
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            MPI.Attach_buffer(Mpi_buf)
            g=[]
            for i in range(0,size-2):
                grad_list=COMM.recv(source=MPI.ANY_SOURCE,tag=999)
                g.append(grad_list)
            g=grad_avg(g)
            optimizer.zero_grad()
            for tmp_g,tmp_p in zip(g, model.named_parameters()):
                if tmp_g is not None:
                    tmp_p[1].grad = tmp_g
            optimizer.step()
            torch.save(model.state_dict(), model_dir)
            param_glob=model.state_dict()
            for i in range(2,size):
                COMM.bsend(param_glob,dest=i,tag=999)    
            MPI.Detach_buffer()   
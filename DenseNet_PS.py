from __future__ import print_function, division

import json
import re
from collections import OrderedDict

import torch.nn.functional as F
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
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
import copy
import argparse

def parse_devices(device_string):
    if device_string is None:
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in device_string.split(',')]

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--Flag', '-f', help='flag of first time to run', required=True)
parser.add_argument('--GPU_list', '-g', help='placement of each process', required=True)
parser.add_argument('--Id_index', '-i', help='index for job (checkpoint)', required=True)
parser.add_argument('--Network', '-n', help='Network:121 169 201 161', required=True)
parser.add_argument('--Batch_size', '-b', help='batch_size:8 16 32 64', required=True)
parser.add_argument('--Epoch', '-e', help='epoch size:40 60 80', required=True)
args = parser.parse_args()
# EPOCH = 20
# if str(args.Epoch)=="10":
#     EPOCH = 10
# elif str(args.Epoch)=="15":
#     EPOCH = 15
# elif str(args.Epoch)=="20":
#     EPOCH = 20

EPOCH=int(str(args.Epoch))
job_id = str(args.Id_index) #The index of job to find the right checkpoint
model_dir = "/root/ExplSched/Scheduling/check_point/" + job_id + "_model.pt"
epoch_dir = "/root/ExplSched/Scheduling/check_point/" + job_id + "_epoch.pt"
#from thop import profile

#from ft.OCI_Iterator import OCIIterator

# from ft.OCI_Iterator import OCIIterator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
'''定义超参数'''

batch_size = 16
if str(args.Batch_size)=="8":
    batch_size = 8
elif str(args.Batch_size)=="16":
    batch_size = 16
elif str(args.Batch_size)=="32":
    batch_size = 32
else:
    batch_size=64
classes_num = 1000
learning_rate = 1e-3


gpu_list = parse_devices(args.GPU_list)
first_run_flag = str(args.Flag)

COMM = MPI.COMM_WORLD
rank=COMM.Get_rank()
size=COMM.Get_size()
DEVICE = torch.device("cuda:0")
device = DEVICE

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


densenet121_model_name = 'densenet121-a639ec97.pth'
densenet169_model_name = 'densenet169-b2777c0a.pth'
densenet201_model_name = 'densenet201-c1103571.pth'
densenet161_model_name = 'densenet161-8d451a50.pth'
models_dir = os.path.expanduser('~/.torch/models')


def densenet121(pretrained=False, **kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(os.path.join(models_dir, densenet121_model_name))
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet169(pretrained=False, **kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(os.path.join(models_dir, densenet169_model_name))
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet201(pretrained=False, **kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(os.path.join(models_dir, densenet201_model_name))
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet161(pretrained=False, **kwargs):
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(os.path.join(models_dir, densenet161_model_name))
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1',
                        nn.Conv2d(num_input_features, bn_size * growth_rate,
                                  kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


# --------------------训练过程---------------------------------
model = densenet201()  # 在这里更换你需要训练的模型
if str(args.Network)=="121":
    model=densenet121()
elif str(args.Network)=="169":
    model=densenet169()
elif str(args.Network)=="161":
    model=densenet161()
elif str(args.Network)=="201":
    model=densenet201()

model_name = r'densenet201'
training_result_dir = r'./training_result'

# summary.summary(model, input_size=(3, 224, 224), device='cuda')  # 我们选择图形的出入尺寸为(3,224,224)

params = [{'params': md.parameters()} for md in model.children()
          if md in [model.classifier]]
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 按训练批次调整学习率，每30个epoch调整一次
# loss_func = nn.CrossEntropyLoss()
loss_func = nn.CrossEntropyLoss().cuda()
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

def parse_devices(device_string):
    if device_string is None:
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in device_string.split(',')]

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

        time_dir = "/root/ExplSched/Scheduling/time_loss/" + "DenseNet"+str(args.Network)+"_"+job_id+"_"+str(rank)+"_time.txt"
        with open(time_dir,"a") as time_t:
            
            if batch_idx % 100 == 0:
                current_t = (time.time() - t) / 60
                epoch_t=(time.time()-start_time)/60
                time_loss=str(epoch)+' '+str(loss.item())+' '+str(current_t)+' '+str(epoch_t)
                time_t.write(time_loss)
                time_t.write('\n')
            
            #     print(f'Epoch: {epoch + 1:03d}/{EPOCH:03d} | '
            #         f'Batch {batch_idx:04d}/{len(train_dataloader):04d} | '
            #         f'Cost: {loss:.4f} | '
            #         f'Time: {current_t:.4f}min')
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


log_dir = './densenet201.pth'
if __name__ == '__main__':
    
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
    
    
    dataset_train_list = distributed_samper(dataset=train_datasets, num_workers=(size-2)*2) 
    dataset_test_list = distributed_samper(dataset=val_datasets, num_workers=(size-2)*2)
    local_batch_size = int(batch_size/((size-2)*2))

    # if first_run_flag == "true": #if the flag is True, represent the MPI process is not the first time to run 
    #         #print("this code run")
        

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
            param_glob=model.state_dict()
            torch.save(model.state_dict(), model_dir)
            for i in range(2,size):
                COMM.bsend(param_glob,dest=i,tag=999)    
            MPI.Detach_buffer()   
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



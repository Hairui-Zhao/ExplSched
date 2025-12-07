# 导入必要的库
import json
import os

import torch
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import time
import math
import copy
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
parser.add_argument('--Network', '-n', help='Network:11 13 16 19', required=True)
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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
'''定义超参数'''

batch_size = 8
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

models_dir = os.path.expanduser('~/.torch/models')
model_name = {
    'vgg11': 'vgg11-bbd30ac9.pth',
    'vgg11_bn': 'vgg11_bn-6002323d.pth',
    'vgg13': 'vgg13-c768596a.pth',
    'vgg13_bn': 'vgg13_bn-abd245e5.pth',
    'vgg16': 'vgg16-397923af.pth',
    'vgg16_bn': 'vgg16_bn-6c64b313.pth',
    'vgg19': 'vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg11'])))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg11_bn'])))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg13'])))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg13_bn'])))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg16'])))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg16_bn'])))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg19'])))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg19_bn'])))
    return model


# 定义训练的辅助函数 包含error与accuracy
def eval(model, loss_func, dataloader):

    model.eval()
    loss, accuracy = 0, 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            logits = model(batch_x)
            error = loss_func(logits, batch_y)
            loss += error.item()

            probs, pred_y = logits.data.max(dim=1)
            accuracy += (pred_y==batch_y.data).float().sum()/batch_y.size(0)

    loss /= len(dataloader)
    accuracy = accuracy*100.0/len(dataloader)
    return loss, accuracy



# def poisson_error_list(ft_lambda):
#     e_list = []
#     e_slot = 0
#     for i in range(10):
#         # r = random.random()
#         r = 0.039
#         e_interval = - math.log(1 - r) / ft_lambda
#         e_slot += e_interval
#         e_list.insert(i, e_slot)
#     return e_list
# error_num = 0
# error_list = poisson_error_list(ft_lambda=0.008)
# error_list[0] = 1.0
# print('error_list:{}'.format(error_list))
# # ck, the unit is 'min'
# OCI = OCIIterator(model_name='vgg19',
#                   dataloader=train_dataloader,
#                   ft_lambda=0.0042,  # 4 hours a error
#                   ck_mode='MANUAL',
#                   ts=0.0016,  # the unit is min
#                   theta_1=-0.241,
#                   theta_2=0.395,
#                   epoch=NUM_EPOCHS,
#                   ft_strategy='CCM',
#                   fit_interval=10000,
#                   profile_threshold=50,
#                   model=model,
#                   optimizer=optimizer)
# ck_t_error = []

model = vgg19_bn()  # 在这里更换你需要训练的模型
if str(args.Network)=="11":
    model=vgg11_bn()
elif str(args.Network)=="13":
    model=vgg13_bn()
elif str(args.Network)=="16":
    model=vgg16_bn()
elif str(args.Network)=="19":
    model=vgg19_bn()

model_Name = r'vgg19_bn'
training_result_dir = r'./training_result'

# summary.summary(model, input_size=(3, 224, 224), device='cuda')  # 我们选择图形的出入尺寸为(3,224,224)

params = [{'params': md.parameters()} for md in model.children()
          if md in [model.classifier]]
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 按训练批次调整学习率，每30个epoch调整一次
# loss_func = nn.CrossEntropyLoss()
loss_func = nn.CrossEntropyLoss().to(DEVICE)
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
    # print(f'len(train_dataloader):{len(train_dataloader)} and len(train_datasets):{len(train_datasets)}')
    # print()
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

        time_dir = "/root/ExplSched/Scheduling/time_loss/" + "VGG"+str(args.Network)+"_"+job_id+"_"+str(rank)+"_time.txt"
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
    
    # Loss_list.append(eval_loss / (len(val_datasets)))
    # Accuracy_list.append(100 * eval_acc / (len(val_datasets)))

log_dir = './vgg19_bn.pth'


BUFFSIZE=2**32
Mpi_buf = bytearray(BUFFSIZE)
if __name__ == '__main__':
    
    

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
        data_loader = DataLoader(dataset_train_list[rank-1], local_batch_size, shuffle=True,num_workers=2,pin_memory=True) 
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
            val_dataloader=DataLoader(dataset_test_list[rank-1], local_batch_size, shuffle=True,num_workers=2,pin_memory=True) 
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
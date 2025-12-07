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


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
'''定义超参数'''
EPOCH = 20
batch_size = 64
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
model_Name = r'vgg19_bn'
training_result_dir = r'./training_result'
model = model.to(DEVICE)
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

        # state = {'model': model.state_dict(),
        #          'optimizer': optimizer.state_dict(),
        #          'epoch': epoch,
        #          'pre_train_t': time.time() - since_t,
        #          'train_time_list': train_time_list,
        #          'train_loss_list': train_loss_list}
        # torch.save(state, './vgg16_bn.pth')


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

log_dir = './vgg19_bn.pth'
def main():
    model.to(DEVICE)

    test_flag = True
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
        f = open(os.path.join(training_result_dir, model_Name + '_time.txt'), 'w+')
        f.write(js)
        f.close()

        js = json.dumps(train_loss_list)
        f = open(os.path.join(training_result_dir, model_Name + '_loss.txt'), 'w+')
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
    f = open(os.path.join(training_result_dir, model_Name + '_time.txt'), 'w+')
    f.write(js)
    f.close()

    js = json.dumps(train_loss_list)
    f = open(os.path.join(training_result_dir, model_Name + '_loss.txt'), 'w+')
    f.write(js)
    f.close()


if __name__ == '__main__':
    main()

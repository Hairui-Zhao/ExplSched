import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
#device = torch.device('cpu')



def test_img(net_g, datatest, gpu_rank):
    if gpu_rank == 0:
        device = torch.device('cuda:0')
    elif gpu_rank == 1:
        device = torch.device('cuda:1')
    elif gpu_rank == 2:
        device = torch.device('cuda:2')
    else:
        device = torch.device('cuda:3')
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=64)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)

    #print('\nTest set: Average loss: {:.4f} '.format(test_loss))
    return accuracy, test_loss
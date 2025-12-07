from mpi4py import MPI 
import pickle
import torch
import copy
import numpy as np
import time
from lib.Avg import *
import os
from torch import nn,optim
from torch.utils.data import DataLoader
from Net.LeNet import *
from lib.test_img import *
import time
from torchvision import datasets, transforms
import sys
from torch.utils.data import random_split
import argparse

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--Flag', '-f', help='flag of first time to run', required=True)
parser.add_argument('--GPU_list', '-g', help='placement of each process', required=True)
parser.add_argument('--Id_index', '-i', help='index for job (checkpoint)', required=True)
parser.add_argument('--Epoch', '-e', help='epoch size:40 60 80', required=True)
args = parser.parse_args()

trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset_train = datasets.MNIST('/root/ExplSched/MNIST', train=True, download=True, transform=trans_mnist)
dataset_test = datasets.MNIST('/root/ExplSched/MNIST', train=False, download=True, transform=trans_mnist)
first_run_flag = str(args.Flag) #The flag to determine whether to change the number of workers
#run_time = int(sys.argv[2]) #Setting the job running time

job_id = str(args.Id_index) #The index of job to find the right checkpoint
model_dir = "/root/ExplSched/Scheduling/check_point/" + job_id + "_model.pt"
epoch_dir = "/root/ExplSched/Scheduling/check_point/" + job_id + "_epoch.pt"
index_dataset = 0
#print(first_run_flag, run_time)
BUFFSIZE=2**30
batch_size = 128
Mpi_buf = bytearray(BUFFSIZE)

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

gpu_list = parse_devices(args.GPU_list)
print(gpu_list, len(gpu_list))

#print(args.GPU_list)
#device = torch.device(gpu_list[0])

if __name__ == '__main__':
    COMM = MPI.COMM_WORLD
    rank=COMM.Get_rank()
    size=COMM.Get_size()
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
    
    dataset_train_list = distributed_samper(dataset=dataset_train, num_workers=(size-2)*2)
    
    dataset_test_list = distributed_samper(dataset=dataset_test, num_workers=(size-2)*2)
   
    local_batch_size = int(batch_size/((size-2)*2))
    
    net = LeNet()
    if first_run_flag == "true": #if the flag is True, represent the MPI process is not the first time to run 
            #print("this code run")
        net.load_state_dict(torch.load(model_dir))
    loss_fun = nn.CrossEntropyLoss()
    epochs = int(str(args.Epoch))
    lr = 0.005
    counter = 0
    epoch = 1
    if rank!=0 and rank!=1:
        net.to(device)
        if first_run_flag == "true": #if the flag is True, represent the MPI process is not the first time to run 
            #print("this code run")
            epoch = torch.load(epoch_dir)
            epoch = epoch + 1
            print(epoch)
        data_loader = DataLoader(dataset_train_list[rank-1], local_batch_size, shuffle=True) 
        t=time.time()
        for epoch in range(epoch, epochs+1):            
            counter += 1
            if counter / 10 == 1:
                counter = 0
                lr = lr * 0.5
            net.train()
            dataloader = iter(data_loader)
            for image, segment_image in dataloader:
                state = epoch
                end_time = time.time()
                #print("end", end_time)
                # if (run_time < (end_time - start_time)):
                #     torch.save(state, "epoch.pt")
                #     sys.exit()
                optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,weight_decay=1e-5)
                data = image.to(device)
                target = segment_image.to(device)
                optimizer.zero_grad()
                output = net(data).to(device)               
                loss = loss_fun(output, target)               
                loss.backward()
                grad_list=[]
                for p in net.parameters():
                    grad=p.grad
                    grad_list.append(grad)
                MPI.Attach_buffer(Mpi_buf)
                COMM.bsend(grad_list,dest=1,tag=999)
                param_glob=COMM.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG)
                MPI.Detach_buffer()
                # if param_glob == "error":
                #     torch.save(state, "epoch.pt")
                #     COMM.bsend(rank,dest=0,tag=1)
                #     MPI.Detach_buffer()
                #     sys.exit(0)
                net.load_state_dict(param_glob)
            time_dir = "/root/ExplSched/Scheduling/time_loss/" + "LeNet_"+job_id+"_"+str(rank)+"_time.txt"
            with open(time_dir,"a") as time_t:
                    current_t = (time.time() - t) / 60
                    time_loss=str(epoch)+' '+str(loss.item())+' '+str(current_t)
                    time_t.write(time_loss)
                    time_t.write('\n')
            net.eval()
            
          #  print( "rank:{:d} \t Training accuracy: {:.6f} \tTest set: Average loss: {:.6f} \tTesting accuracy: {:.6f}".format(rank,acc_train,loss_train,acc_test))
            torch.save(epoch, epoch_dir,_use_new_zipfile_serialization=False)
            torch.save(net.state_dict(), model_dir)
            
    elif rank==1:
        net.to(device)
        if first_run_flag == "true": #if the flag is True, represent the MPI process is not the first time to run 
            #print("this code run")
            net.load_state_dict(torch.load(model_dir))
        
        while(1):
            # label = []
            # for i in range(0,size-1):
            #     label_list = COMM.recv(source=MPI.ANY_SOURCE,tag=1)
            #     label.append(label_list)
            # if len(label) == size-1:
            #     sys.exit()
            #print("###")
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,weight_decay=1e-5)
            MPI.Attach_buffer(Mpi_buf)
            g=[]
            
            for i in range(0,size-2):
                grad_list=COMM.recv(source=MPI.ANY_SOURCE,tag=999)
                
                #label_list = int(label_list)
                #if label_list == 1 or label_list == 2 or label_list == 3 or label_list == 4:
                g.append(grad_list)
            g=grad_avg(g)
            optimizer.zero_grad()
            #print("grad", g)
            #print("net", net.named_parameters())
            for tmp_g,tmp_p in zip(g, net.named_parameters()):
                if tmp_g is not None:
                    tmp_p[1].grad = tmp_g
            optimizer.step()
            param_glob=net.state_dict()
            for i in range(2,size):
                COMM.bsend(param_glob,dest=i,tag=999)
            
            MPI.Detach_buffer()




                
        
import time
import os
import subprocess
import sys
import argparse
import signal
import psutil
import random
import torch

class Job():
    ID=''
    name=''
    flag=''
    net=''
    batch_size=''
    epoch=''
    importance=0    #SLAQ特有属性
    GPU_possess=[]
    GPU_list=[]
    start_time=0
    end_time=0

    def __init__(self,id,nm,fg,nt,bs,ep,im):
        self.ID=id
        self.name=nm
        self.flag=fg
        self.net=nt
        self.batch_size=bs
        self.epoch=ep
        self.GPU_possess=[]
        self.GPU_list=[]
        self.start_time=0
        self.end_time=0
        self.importance=int(im)

exe_queue={}
wait_queue=[]
finish_queue=[]
GPU_state=[0,0,0,0]
SCHEDUL_INTERVAL=600

def kill_process(pid):
    parent_proc = psutil.Process(pid)
    for child_proc in parent_proc.children(recursive=True):
        child_proc.kill()
    parent_proc.kill() 

def handle_jobs():
    global wait_queue
    # 读取job序列文件，将job序列转化为job_list#########################
    with open("/root/ExplSched/Scheduling/run/job.txt","a+") as jobs:
        j=jobs.read()
        t1=time.time()
        t2=time.time()
        while(j==''and t2-t1<10):
            jobs.seek(0)
            j=jobs.read()
            t2=time.time()
        jobs.seek(0)
        jobs.truncate(0)
    j=j.split('\n')
    j.pop()
    for job in j:
        msg=job.split(' ')
        print(msg)
        temp_job=Job(msg[0],msg[1],msg[2],msg[3],msg[4],msg[5],msg[6])
        temp_job.start_time=time.time()
        wait_queue.append(temp_job)

def assign_resource(job):
    global wait_queue
    global exe_queue
    global finish_queue
    global GPU_state
    if (job.name=="de" and job.net=="169") or (job.name=="re" and job.net=="152") or(job.name=="de" and job.net=="201") or(job.name=="in"):
        if GPU_state.count(0)>=4:
            idx0=GPU_state.index(0)
            GPU_state[idx0]=1
            idx1=GPU_state.index(0)
            GPU_state[idx1]=1
            idx2=GPU_state.index(0)
            GPU_state[idx2]=1
            idx3=GPU_state.index(0)
            GPU_state[idx3]=1
            job.GPU_list.append(idx0)
            job.GPU_list.append(idx0)
            job.GPU_list.append(idx0)
            job.GPU_list.append(idx1)
            job.GPU_list.append(idx2)
            job.GPU_list.append(idx3)
            job.GPU_possess.append(idx0)
            job.GPU_possess.append(idx1)
            job.GPU_possess.append(idx2)
            job.GPU_possess.append(idx3)
        else:
            return 0
    elif (job.name=="vg" and job.net=="13") or (job.name=="vg" and job.net=="19"):
        if GPU_state.count(0)>=3:
            idx0=GPU_state.index(0)
            GPU_state[idx0]=1
            idx1=GPU_state.index(0)
            GPU_state[idx1]=1
            idx2=GPU_state.index(0)
            GPU_state[idx2]=1
            job.GPU_list.append(idx0)
            job.GPU_list.append(idx0)
            job.GPU_list.append(idx0)
            job.GPU_list.append(idx1)
            job.GPU_list.append(idx2)

            job.GPU_possess.append(idx0)
            job.GPU_possess.append(idx1)
            job.GPU_possess.append(idx2)

        else:
            return 0
    elif (job.name=="re" and job.net=="50") or (job.name=="vg" and job.net=="11"):
        if GPU_state.count(0)>=2:
            idx0=GPU_state.index(0)
            GPU_state[idx0]=1
            idx1=GPU_state.index(0)
            GPU_state[idx1]=1
            job.GPU_list.append(idx0)
            job.GPU_list.append(idx0)
            job.GPU_list.append(idx0)
            job.GPU_list.append(idx1)

            job.GPU_possess.append(idx0)
            job.GPU_possess.append(idx1)

        else:
            return 0
    elif (job.name=="le") or (job.name=="re" and job.net=="18"):
    
        if GPU_state.count(0)>=1:
            idx0=GPU_state.index(0)
            GPU_state[idx0]=1
            job.GPU_list.append(idx0)
            job.GPU_list.append(idx0)
            job.GPU_list.append(idx0)

            job.GPU_possess.append(idx0)
        else:
            return 0
    # elif  :
    #     if GPU_state.count(0)>=2:
    #         idx0=GPU_state.index(0)
    #         GPU_state[idx0]=1
    #         idx1=GPU_state.index(0)
    #         GPU_state[idx1]=1

    #         job.GPU_list.append(idx0)
    #         job.GPU_list.append(idx0)
    #         job.GPU_list.append(idx1)

    #         job.GPU_possess.append(idx0)
    #         job.GPU_possess.append(idx1)

    #     else:
    #         return 0
    # elif job.name=="in":
    #     if GPU_state.count(0)>=2:
    #         idx0=GPU_state.index(0)
    #         GPU_state[idx0]=1
    #         idx1=GPU_state.index(0)
    #         GPU_state[idx1]=1
    #         job.GPU_list.append(idx0)
    #         job.GPU_list.append(idx0)
    #         job.GPU_list.append(idx1)
    #         job.GPU_possess.append(idx0)
    #         job.GPU_possess.append(idx1)
    #     else:
    #         return 0
    # elif job.name=="re" and job.net=="152":
    #     if GPU_state.count(0)>=2:
    #         idx0=GPU_state.index(0)
    #         GPU_state[idx0]=1
    #         idx1=GPU_state.index(0)
    #         GPU_state[idx1]=1
    #         job.GPU_list.append(idx0)
    #         job.GPU_list.append(idx0)
    #         job.GPU_list.append(idx1)
    #         job.GPU_possess.append(idx0)
    #         job.GPU_possess.append(idx1)
    #     else:
    #         return 0
    return 1

def sw(l):
    for i in range(len(l)):
        l[i]=str(l[i])
    return l

def execute_job(job):
    global wait_queue
    global exe_queue
    global finish_queue
    global GPU_state
    res=",".join(sw(job.GPU_list))
    epoch_dir="/root/ExplSched/Scheduling/check_point/"+job.ID+"_epoch.pt"
    if os.path.exists(epoch_dir):
        job.flag="true"
    else:
        job.flag="false"
    if job.name=="vg":
        p=subprocess.Popen("mpirun -np {:d} python ../../VGG_PS.py -f {:s} -g {:s} -i {:s} -n {:s} -b {:s} -e {:s}"
                                .format(int(len(job.GPU_list)),job.flag,res,job.ID,job.net,job.batch_size,job.epoch),shell=True)
    if job.name=="de":
        p=subprocess.Popen("mpirun -np {:d} python ../../DenseNet_PS.py -f {:s} -g {:s} -i {:s} -n {:s} -b {:s} -e {:s}"
                                .format(int(len(job.GPU_list)),job.flag,res,job.ID,job.net,job.batch_size,job.epoch),shell=True)
    if job.name=="re":
        p=subprocess.Popen("mpirun -np {:d} python ../../ResNet_Image_PS.py -f {:s} -g {:s} -i {:s} -n {:s} -b {:s} -e {:s}"
                                .format(int(len(job.GPU_list)),job.flag,res,job.ID,job.net,job.batch_size,job.epoch),shell=True)
    if job.name=="in":
        p=subprocess.Popen("mpirun -np {:d} python ../../Inception_v4_PS.py -f {:s} -g {:s} -i {:s} -e {:s}"
                                .format(int(len(job.GPU_list)),job.flag,res,job.ID,job.epoch),shell=True)
    if job.name=="le":
        p=subprocess.Popen("mpirun -np {:d} python ../LeNet_PS.py -f {:s} -g {:s} -i {:s} -e {:s}"
                                .format(int(len(job.GPU_list)),job.flag,res,job.ID,job.epoch),shell=True)
    exe_queue[job]=p


def is_finish(job):
    global wait_queue
    global exe_queue
    global finish_queue
    global GPU_stat
    epoch_dir="/root/ExplSched/Scheduling/check_point/"+job.ID+"_epoch.pt"
    if os.path.exists(epoch_dir):
        cur_epoch=torch.load(epoch_dir)
        if int(cur_epoch)!=int(job.epoch):
            return 0
        else:
            job.end_time=time.time()
            Id=exe_queue[job].pid
            kill_process(Id)
            finish_queue.append(job)
            print("job{:s} is finished".format(exe_job.ID))
            for idx in job.GPU_possess:
                GPU_state[idx]=0
            return 1
    else:
        return 0

def sort_job(job_list):
    job_list=sorted(job_list,key=lambda Job:Job.importance,reverse=True)
    return job_list

xxxtime=time.time()
handle_jobs()

while len(finish_queue)<20:
    t1=time.time()
    t2=time.time()
    temp_f=[]
    #print(wait_queue)
    wait_queue=sort_job(wait_queue)
    #print(wait_queue)
    for wait_job in wait_queue:
        if assign_resource(wait_job)==1:
            execute_job(wait_job)
    while t2-t1<=SCHEDUL_INTERVAL: #调度间
        handle_jobs()
        time.sleep(1)
        t2=time.time()
    temp=[]
    for exe_job in exe_queue:
        if is_finish(exe_job)==1:
            temp_f.append(exe_job)
        else:
            Id=exe_queue[exe_job].pid
            kill_process(Id)
            print("job{:s} is killed".format(exe_job.ID))
        temp.append(exe_job)
    for exe_job in temp:
        exe_queue.pop(exe_job)
        for idx in exe_job.GPU_possess:
            GPU_state[idx]=0
        exe_job.GPU_list.clear()
        exe_job.GPU_possess.clear()
        exe_job.flag="true"
        
    for finish_job in temp_f:
        wait_queue.remove(finish_job)

with open("/root/ExplSched/Scheduling/run/JCT.txt","w+") as Jct:
    for finish_job in finish_queue:
        f_t=(finish_job.end_time-finish_job.start_time)/60
        msg=finish_job.ID+" "+str(f_t)+" "+str(finish_job.start_time-xxxtime)+" "+str(finish_job.end_time-xxxtime)
        Jct.write(msg)
        Jct.write("\n")
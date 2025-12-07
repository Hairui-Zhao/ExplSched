import time
import os
import subprocess
import sys
import argparse
import signal
import psutil
import random
import torch

JOB_NUM=0

class Job():
    ID=''
    name=''
    flag=''
    net=''
    batch_size=''
    epoch=''
    target_epoch='' #Expl特有属性
    GPU_possess=[]
    GPU_list=[]
    start_time=0
    end_time=0
    achieve_time=0

    def __init__(self,id,nm,fg,nt,bs,ep,t_ep):
        self.ID=id
        self.name=nm
        self.flag=fg
        self.net=nt
        self.batch_size=bs
        self.epoch=ep
        self.target_epoch=t_ep
        self.GPU_possess=[]
        self.GPU_list=[]
        self.start_time=0
        self.end_time=0
        self.achieve_time=0

exe_queue={}
wait_queue=[]
achieve_queue=[]
finish_queue=[]
GPU_state=[0,0,0,0]

def kill_process(pid):
    parent_proc = psutil.Process(pid)
    for child_proc in parent_proc.children(recursive=True):
        child_proc.kill()
    parent_proc.kill()

def handle_jobs():
    global JOB_NUM
    global wait_queue
    # 读取job序列文件，将job序列转化为job_list#########################
    with open("/root/ExplSched/Scheduling/run/job.txt","r+") as jobs:
        j=jobs.read()
        t1=time.time()
        t2=time.time()
        while(j=='' and t2-t1<5):
            jobs.seek(0)
            j=jobs.read()
            t2=time.time()
        jobs.seek(0)
        jobs.truncate(0)
    if j!='':
        j=j.split('\n')
        j.pop()
        for job in j:
            msg=job.split(' ')
            print(msg)
            temp_job=Job(msg[0],msg[1],msg[2],msg[3],msg[4],msg[5],msg[6])
            wait_queue.append(temp_job)
            JOB_NUM=JOB_NUM+1

def assign_resource(job):
    global exe_queue
    global wait_queue
    global finish_queue
    global achieve_queue
    global GPU_state
    # for i in range(8):
    #     if i==0:
    #         job.GPU_list.append(i)
    #     job.GPU_list.append(i)
    #     job.GPU_possess.append(i)
    if job.name=="le":
        if GPU_state.count(0)>=1:
            idx0=GPU_state.index(0)
            GPU_state[idx0]=1
            job.GPU_list.append(idx0)
            job.GPU_list.append(idx0)
            job.GPU_list.append(idx0)
            job.GPU_possess.append(idx0)
        else:
            return 0
    elif job.name=="re" and job.net=="18":
        if GPU_state.count(0)>=1:
            idx0=GPU_state.index(0)
            GPU_state[idx0]=1
            job.GPU_list.append(idx0)
            job.GPU_list.append(idx0)
            job.GPU_list.append(idx0)
            job.GPU_possess.append(idx0)
        else:
            return 0
    elif (job.name=="re" and job.net=="50")or (job.name=="vg" and job.net=="11")or  (job.name=="vg" and job.net=="13"):
        if GPU_state.count(0)>=2:
            idx0=GPU_state.index(0)
            GPU_state[idx0]=1
            job.GPU_list.append(idx0)
            job.GPU_list.append(idx0)
            job.GPU_list.append(idx0)

            job.GPU_possess.append(idx0)

        else:
            return 0
    elif (job.name=="de" and job.net=="169") or (job.name=="de" and job.net=="201") or (job.name=="vg" and job.net=="19") or(job.name=="in")or(job.name=="re" and job.net=="152"):
        if GPU_state.count(0)>=4:
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
    return 1

def assign_resource_f(job):
    global wait_queue
    global exe_queue
    global finish_queue
    global GPU_state
    if GPU_state.count(0)>=1:
        # idx0=GPU_state.index(0)
        # GPU_state[idx0]=1
        # idx1=GPU_state.index(0)
        # GPU_state[idx1]=1
        # job.GPU_list.append(idx0)
        # job.GPU_list.append(idx0)
        # job.GPU_list.append(idx1)
        # job.GPU_possess.append(idx0)
        # job.GPU_possess.append(idx1)
        idx0=GPU_state.index(0)
        GPU_state[idx0]=1
        job.GPU_list.append(idx0)
        job.GPU_list.append(idx0)
        job.GPU_list.append(idx0)
        job.GPU_possess.append(idx0)
        return 1
    else:
        return 0 

def is_finish(job):
    global exe_queue
    global wait_queue
    global finish_queue
    global achieve_queue
    global GPU_state
    epoch_dir="/root/ExplSched/Scheduling/check_point/"+job.ID+"_epoch.pt" #需要修改
    cur_epoch=torch.load(epoch_dir)
    if int(cur_epoch)!=int(job.epoch):
        return 0
    else:
        job.end_time=time.time()
        Id=exe_queue[job].pid
        kill_process(Id)
        print("job{:s} is finished!!!!!!!!".format(job.ID))
        finish_queue.append(job)
        for idx in job.GPU_possess:
            GPU_state[idx]=0
        return 1

def is_achieve(job):
    global exe_queue
    global wait_queue
    global finish_queue
    global achieve_queue
    global GPU_state
    epoch_dir="/root/ExplSched/Scheduling/check_point/"+job.ID+"_epoch.pt" #需要修改
    if os.path.exists(epoch_dir):
        cur_epoch=torch.load(epoch_dir)

        if int(cur_epoch)<int(job.target_epoch):
            return 0
        else:
            Id=exe_queue[job].pid
            kill_process(Id)
            job.achieve_time=time.time()
            print("job{:s} is achieved!!!!!!!!".format(job.ID))
            for idx in job.GPU_possess:
                GPU_state[idx]=0
            job.GPU_list.clear()
            job.GPU_possess.clear()
            job.flag="true"
            
            if int(job.target_epoch)==int(job.epoch):
                job.end_time=time.time()
                print("job{:s} is finished!!!!!!!!".format(job.ID))
                finish_queue.append(job)
                return 2
            else:
                achieve_queue.append(job)
                return 1
    else:
        return 0

def sw(l):
    for i in range(len(l)):
        l[i]=str(l[i])
    return l

def execute_job(job):
    global exe_queue
    global wait_queue
    global finish_queue
    global achieve_queue
    global GPU_state
    res=",".join(sw(job.GPU_list)) 
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
    
    if job.flag=="false":
        job.start_time=time.time()

xxxtime=time.time()
while(1):
    if JOB_NUM < 20:
        handle_jobs()
        temp_wait=[]
        for wait_job in wait_queue:
            if assign_resource(wait_job)==1:
                execute_job(wait_job)
                temp_wait.append(wait_job)
        for wait_job in temp_wait:
            wait_queue.remove(wait_job)
        temp=[] 
        flag=0
        for exe_job in exe_queue:
            ver=is_achieve(exe_job)
            if ver==0:
                continue
            else:
                temp.append(exe_job)
                flag=1
        if flag==1:
            for temp_job in temp:
                exe_queue.pop(temp_job)
    else:
        while(1):
            time.sleep(1)
            temp_wait=[]
            for wait_job in wait_queue:
                #print("jobid:{:s}".format(wait_job.ID))
                if assign_resource(wait_job)==1:
                    #print("ass jobid:{:s}".format(exe_job.ID))
                    execute_job(wait_job)
                    temp_wait.append(wait_job)
            for wait_job in temp_wait:
                wait_queue.remove(wait_job)
            temp=[] 
            flag=0
            for exe_job in exe_queue:
                #print("jobid:{:s}".format(exe_job.ID))
                if is_achieve(exe_job)==0:
                    continue
                else:
                    temp.append(exe_job)
                    flag=1   
            if flag==1:
                for temp_job in temp:
                    exe_queue.pop(temp_job)
            if len(wait_queue)==0:
                break

        while len(exe_queue)>0:
            time.sleep(1)
            temp=[] 
            flag=0
            for exe_job in exe_queue:
                if is_achieve(exe_job)==0:
                    continue
                else:
                    temp.append(exe_job)
                    flag=1
            if flag==1:
                for temp_job in temp:
                    exe_queue.pop(temp_job)

        while(1):
            temp_achieve=[]
            time.sleep(1)
            for achieve_job in achieve_queue:
                if assign_resource_f(achieve_job)==1:
                    execute_job(achieve_job)
                    temp_achieve.append(achieve_job)   
            for achieve_job in temp_achieve:
                achieve_queue.remove(achieve_job)
            temp=[] 
            flag=0
            for exe_job in exe_queue:
                if is_finish(exe_job)==0:
                    continue
                else:
                    temp.append(exe_job)
                    flag=1   
            if flag==1:
                for temp_job in temp:
                    exe_queue.pop(temp_job)
            if len(achieve_queue)==0:
                break

        while len(exe_queue)>0:
            temp=[] 
            flag=0
            time.sleep(1)
            for exe_job in exe_queue:
                if is_finish(exe_job)==0:
                    continue
                else:
                    temp.append(exe_job)
                    flag=1
            if flag==1:
                for temp_job in temp:
                    exe_queue.pop(temp_job)
        break

with open("/root/ExplSched/Scheduling/run/JCT.txt","w+") as Jct:
    for finish_job in finish_queue:
        f_t=(finish_job.end_time-finish_job.start_time)/60
        msg=finish_job.ID+" "+str(f_t)+" "+str(finish_job.start_time-xxxtime)+" "+str(finish_job.achieve_time-xxxtime)+" "+str(finish_job.end_time-xxxtime)
        Jct.write(msg)
        Jct.write("\n")
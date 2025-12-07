import time
import os
import subprocess
import sys
import argparse
import signal
import psutil
import random
import torch
import copy

JOB_NUM=0

class Job():
    ID=''
    name=''
    flag=''
    net=''
    batch_size=''
    epoch=''
    GPU_possess=[]
    GPU_list=[]
    start_time=0
    end_time=0

    def __init__(self,id,nm,fg,nt,bs,ep):
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
exe_queue={}
wait_queue=[]
new_queue=[]
finish_queue=[]
GPU_state=[0,0,0,0]

def kill_process(pid):
    parent_proc = psutil.Process(pid)
    for child_proc in parent_proc.children(recursive=True):
        child_proc.kill()
    parent_proc.kill()

def handle_jobs():
    global JOB_NUM
    global new_queue
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
            temp_job=Job(msg[0],msg[1],msg[2],msg[3],msg[4],msg[5])
            new_queue.append(temp_job)
            JOB_NUM=JOB_NUM+1
def sw(l):
    r = []
    for i in range(len(l)):
        r.append(str(l[i]))
    return r

def execute_job(job):
    global exe_queue
    global wait_queue
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
    if job.flag=="false":
        job.start_time=time.time()


def is_finish(job):
    global exe_queue
    global wait_queue
    global finish_queue
    global GPU_state
    epoch_dir="/root/ExplSched/Scheduling/check_point/"+job.ID+"_epoch.pt"
    if os.path.exists(epoch_dir):
        cur_epoch=torch.load(epoch_dir)
        if int(cur_epoch)!=int(job.epoch):
            return 0
        else:
            job.end_time=time.time()
            Id=exe_queue[job].pid
            kill_process(Id)
            print("job{:s} is finished".format(job.ID))
            finish_queue.append(job)
            for idx in job.GPU_possess:
                GPU_state[idx]=0
            return 1
    else:
        return 0

xxxtime=time.time()
while(1):
    if JOB_NUM <20:
        handle_jobs()
        for job in new_queue:
            temp=[]
            for exe_job in exe_queue:
                if is_finish(exe_job)==0:
                    continue
                else:
                    temp.append(exe_job)
                    break
            if len(temp)>0:
                exe_queue.pop(temp[0])
                
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
                temp=[]
                seed=random.randint(1,len(exe_queue))
                print(seed)
                loc=0
                for exe_job in exe_queue:
                    loc=loc+1
                    if loc==seed:
                        Id=exe_queue[exe_job].pid
                        kill_process(Id)
                        print("job{:s} is killed".format(exe_job.ID))
                        temp.append(exe_job)
                        break
                for exe_job in temp:
                    exe_queue.pop(exe_job)
                    #print(exe_job.GPU_list)
                    for i in exe_job.GPU_list:
                        job.GPU_list.append(i)
                    for i in exe_job.GPU_possess:
                        job.GPU_possess.append(i)
                    print(job.GPU_list)
                    #print("GGGGGGGGGGG")
                    time.sleep(1)
                    exe_job.GPU_list.clear()
                    exe_job.GPU_possess.clear()
                    exe_job.flag="true"
                    wait_queue.append(exe_job)        
              
            execute_job(job)
            
        new_queue.clear()
    else:
        while(1):
            time.sleep(1)
            if len(exe_queue)==0:
                break
            temp=[]
            for exe_job in exe_queue:
                if is_finish(exe_job)==0:
                    continue
                else:
                    temp.append(exe_job)
                    break
            if len(temp)>0:
                exe_queue.pop(temp[0])
            
            if len(wait_queue)>0 and len(temp)>0 :
                temp_j=[]
                for wait_job in wait_queue:
                    wait_job.GPU_possess=copy.deepcopy(temp[0].GPU_possess)
                    wait_job.GPU_list=copy.deepcopy(temp[0].GPU_list)
                    execute_job(wait_job)
                    temp_j.append(wait_job)
                    break
                for jn in temp_j:
                    wait_queue.remove(jn)
            if len(finish_queue)==20:
                break
        break
            
with open("/root/ExplSched/Scheduling/run/JCT.txt","w+") as Jct:
    for finish_job in finish_queue:
        f_t=(finish_job.end_time-finish_job.start_time)/60
        msg=finish_job.ID+" "+str(f_t)+" "+str(finish_job.start_time-xxxtime)+" "+str(finish_job.end_time-xxxtime)
        Jct.write(msg)
        Jct.write("\n")
        

           


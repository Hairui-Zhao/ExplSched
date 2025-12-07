import time
import random
net_list=["0 vg false 13 32 3 2","1 le false 0 128 10 6","2 re false 18 64 15 5"]
with open("/root/ExplSched/Scheduling/run/job.txt","r+") as job:
    for i in range(20):
        job.seek(0)
        if i==0:
            str="0 de false 169 32 4 1"# 5  
        if i==1:
            str="1 re false 18 64 6 2" # 3
        if i==2:
            str="2 re false 18 64 6 2" # 3
        if i==3:
            str="3 vg false 11 64 6 1" # 5
        if i==4:
            str="4 le false 0 128 6 2" # 2  
        if i==5:
            str="5 de false 201 32 8 1" # 4
        if i==6:
            str="6 le false 0 128 6 2" # 2
        if i==7:
            str="7 re false 50 64 6 2" # 4
        if i==8:
            str="8 le false 0 128 6 2" # 2
        if i==9:
            str="9 vg false 13 32 6 2"  # 6  
        if i==10:
            str="10 le false 0 128 6 2" # 2
        if i==11:
            str="11 vg false 19 16 4 1" # 6
        if i==12:
            str="12 in false 0 64 6 1" # 7
        if i==13:
            str="13 re false 152 32 8 1" # 8
        if i==14:
            str="14 re false 50 64 6 2" # 4
        if i==15:
            str="15 re false 18 32 6 2" # 3
        if i==16:
            str="16 le false 0 128 6 2" # 2 
        if i==17:
            str="17 de false 169 32 6 1" # 5
        if i==18:
            str="18 in false 0 64 6 1" # 7 
        if i==19:
            str="19 vg false 13 32 6 1" # 6
        job.write(str)
        print("generate job")
        job.write('\n')
        job.flush()
        time.sleep(90)
        



        
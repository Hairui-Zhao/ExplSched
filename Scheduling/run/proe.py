with open("/root/ExplSched/Scheduling/run/JCT.txt","r+") as jobs:
    j=jobs.read()
    j=j.split('\n')
    for i in range(len(j)):
        j[i]=j[i].split(' ')
        print(j[i][0])
    fixed_time = float(j[4][2]) 
    for i in range(20):
        j[i][1] = float(j[i][1])*2.5
        j[i][2] = float(j[i][2])*2.5
        j[i][3] = float(j[i][3])*2.5
        j[i][4] = float(j[i][4])*2.5
    print(j)
def sw(l):
    for i in range(len(l)):
        l[i]=str(l[i])
    return l

with open("/root/ExplSched/Scheduling/run/JCT1.txt","w+") as job1s:
    for finish_job in j:
        res=" ".join(sw(finish_job))
        job1s.write(res)
        job1s.write("\n")
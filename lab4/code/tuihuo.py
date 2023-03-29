import random
import numpy as np
import pandas as pd
from math import e
from math import exp
import matplotlib.pyplot as plt
import time

# 载入数据
df = pd.read_csv('kroA100.tsp', skiprows=5, delimiter="\t")
# city = np.array(df[0][0:len(df)-2])  # 最后一行为EOF，不读入
# city_name = df.values.tolist()
# city=city_name[0]
# city_name=df.values.split()
# city_name=city_name[1][0]
# print(city_name)
location = []
df = df.values
df = df[0:-1]
for i in range(len(df)):
    tcity = df[i][0].split()
    city_x = int(tcity[1])
    city_y = int(tcity[2])
    city_location = []
    city_location.append(city_x)
    city_location.append(city_y)
    location.append(city_location)
    # print(city_location)
pc=0.5
Tend=0.5
Tstart=5000

#两个城市的距离
def dist(a, b):
    distance = ((location[a][0] - location[b][0]) ** 2 + (location[a][1] - location[b][1]) ** 2) ** 0.5
    return distance
#路程总长
def sumval(a):
    sum = 0
    for i in range(len(a)-1):
        sum += dist(a[i], a[i+1])
    sum+=dist(a[len(a)-1],a[0])
    return sum

#产生新解,可以改动
def creat_new(former):
    newway = former.copy()
    a = random.randint(0,len(location)-1)
    b = random.randint(0,len(location)-1)
    newway[a], newway[b] = newway[b], newway[a]
    return newway

def create_new2(former):
    loc1,loc2,loc3=np.random.randint(0, len(location)-1, 3)
    # 下面的三个判断语句使得loc1<loc2<loc3
    if loc1 > loc2:
        loc1, loc2 = loc2, loc1
    if loc2 > loc3:
        loc2, loc3 = loc3, loc2
    if loc1 > loc2:
        loc1, loc2 = loc2, loc1
    # 将[loc1,loc2)区间的数据插入到loc3之后
    newways=former.copy()
    changepart = newways[loc1:loc2].copy()
    newways[loc1:loc3 - loc2 + 1 + loc1] = newways[loc2:loc3 + 1].copy()
    newways[loc3 - loc2 + 1 + loc1:loc3 + 1] = changepart.copy()
    return newways


#还可以改下T的变化函数
def printfig(ans,T,starttime):
    #plt.cla()表示清除当前轴
    #plt.clf()表示清除当前数字
    #plt.ion()用于打开交互模式
    #plt.ioff()用于关闭交互模式
    plt.cla()
    plt.clf()
    X = []
    Y = []
    for i in range(len(ans)):
        x = location[ans[i]][0]
        y = location[ans[i]][1]
        X.append(x)
        Y.append(y)
    x = location[ans[0]][0]
    y = location[ans[0]][1]
    X.append(x)
    Y.append(y)
    plt.scatter(x, y)
    plt.plot(X, Y, '-o')
    plt.title("temp answer:")
    #plt.savefig('./img/pic-{}.png'.format(printnum + 1))
    plt.show()
    plt.pause(0.5)
    if T<=Tend:
        plt.ioff()
        endtime = time.perf_counter()
        print('Running time: %s Seconds' % (endtime - starttime))
        plt.show()



def main():
    starttime = time.perf_counter()
    tempway=[]
    for i in range(len(location)):
        tempway.append(i)
    T = Tstart
    cnt = 0
    trend = []
    while T > Tend:
        for i in range(1000):
            if np.random.random()<pc:
                newway = creat_new(tempway)
            else:
                newway =create_new2(tempway)
            old_dist = sumval(tempway)
            new_dist = sumval(newway)
            p=0
            if new_dist - old_dist < 0:
                p=1
            else:
                p=exp(-(new_dist - old_dist) / T)
            if np.random.random() < p:
                tempway = newway
        T = T * 0.98
        cnt += 1
        bestlen=sumval(tempway)
        trend.append(bestlen)
        print(cnt,"次降温，温度为：",T," 路程长度为：", bestlen)
        plt.ion()
        if T<=Tend or cnt%5==0:
            printfig(tempway, T,starttime)
    lenfinal=sumval(tempway)
    print("最终路程长度为",lenfinal)
    print("最终路线为",tempway)
    plt.plot(trend)
    plt.show()



if __name__ == '__main__':
    main()



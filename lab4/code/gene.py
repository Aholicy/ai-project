import random
import numpy as np
import pandas as pd
from math import e
from math import exp
import matplotlib.pyplot as plt
import time
df = pd.read_csv('kroA100.tsp', skiprows=5, delimiter="\t")
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

# 变异概率
pm1 = 0.2
pm2 = 0.3
pm3 = 0.3
pc = 0.6
parent_size = 150
group_size = 500
trying_times = 1200

def printfig(ans, printnum,starttime):
    # plt.cla()表示清除当前轴
    # plt.clf()表示清除当前数字
    # plt.ion()用于打开交互模式
    # plt.ioff()用于关闭交互模式
    plt.cla()
    plt.clf()
    X = []
    Y = []
    for i in range(len(ans)):
        x = location[ans[i]][0]
        y = location[ans[i]][1]
        X.append(x)
        Y.append(y)
    x=location[ans[0]][0]
    y=location[ans[0]][1]
    X.append(x)
    Y.append(y)
    plt.scatter(x, y)
    plt.plot(X, Y, '-o')
    # plt.savefig('./img/pic-{}.png'.format(printnum + 1))
    plt.show()
    plt.pause(0.1)
    if printnum >= trying_times:
        plt.ioff()
        endtime = time.perf_counter()
        print('Running time: %s Seconds' % (endtime - starttime))
        plt.show()

def distance(a, b):
    distanceance = ((location[a][0] - location[b][0]) ** 2 + (location[a][1] - location[b][1]) ** 2) ** 0.5
    return distanceance


def init_group(group_size):
    races = []
    route = []
    for i in range(len(location)):
        route.append(i)
    for i in range(group_size):
        newroute = route.copy()
        random.shuffle(newroute)
        races.append(newroute)
    return races


def sumval(a):
    sum = 0
    for i in range(len(a) - 1):
        sum += distance(a[i], a[i + 1])
    sum += distance(a[len(a) - 1], a[0])
    return sum


def suitfunc(a):
    sum = sumval(a)
    return (1 / sum) ** 15


def groupfunc(group):
    funclist = []
    # print(len(group))
    for i in range(len(group)):
        temp = suitfunc(group[i])
        funclist.append(temp)
    return funclist


def sort2(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    arrIndex = np.array(arr2).argsort()
    arr1 = arr1[arrIndex]
    arr2 = arr2[arrIndex]
    arr1.tolist()
    arr2.tolist()


def tournament(old_group, size):
    parents = []
    for i in range(size):
        s1, s2 = np.random.randint(0, len(old_group), 2)
        if sumval(old_group[s1]) > sumval(old_group[s2]):
            parents.append(old_group[s2])
        else:
            parents.append(old_group[s1])
    return parents


def nextgene(group, parent):
    new_races = []
    while len(new_races) < group_size:
        i1, i2, i3 = np.random.randint(0, len(parent), 3)
        p1 = parent[i1].copy()
        p2 = parent[i2].copy()
        p3 = parent[i3].copy()
        # 进行交叉
        if np.random.random() < pc:
            p4 = CPX(p1, p2)
            new_races.append(p4)
        else:
            p5, p6 = cross(p1, p2)
            new_races.append(p5)
            new_races.append(p6)
        # 概率小于pm就进行变异
        if np.random.random() < pm1:
            # 变异
            p1 = variation(p1)
        if np.random.random() < pm2:
            p2 = inversion_mutation(p2)
        if np.random.random() < pm3:
            p3 = slide_mutation(p3)
        new_races.extend([p1, p2, p3])
    # sort2(new_races,groupfunc(new_races))
    #final_race = []
    #for i in range(len(group) - parent_size):
        #final_race.append(new_races[i])
    final_race = tournament(new_races, group_size - parent_size)
    for i in range(parent_size):
        final_race.append(parent[i])
    return final_race


def CPX(parent1, parent2):
    cycle = []  # 交叉点集
    start = parent1[0]
    cycle.append(start)
    end = parent2[0]
    while end != start:
        cycle.append(end)
        end = parent2[parent1.index(end)]
    child = parent1[:]
    cross_points = cycle[:]
    if len(cross_points) < 2:
        cross_points = random.sample(parent1, 2)
    k = 0
    for i in range(len(parent1)):
        if child[i] in cross_points:
            continue
        else:
            for j in range(k, len(parent2)):
                if parent2[j] in cross_points:
                    continue
                else:
                    child[i] = parent2[j]
                    k = j + 1
                    break
    return child


def cross(parent1, parent2):
    a = random.randint(1, len(location) - 1)
    child1, child2 = parent1[:a], parent2[:a]

    for i in range(len(location)):
        if parent2[i] not in child1:
            child1.append(parent2[i])

        if parent1[i] not in child2:
            child2.append(parent1[i])

    return child1, child2


# 变异
def variation(path):
    i1, i2, = random.sample(range(0, len(location) - 1), 2)
    path[i1], path[i2] = path[i2], path[i1]
    return path


def inversion_mutation(path):
    a, b = random.sample(range(0, len(location) - 1), 2)
    for i in range(a, (a + b) // 2 + 1):
        path[i], path[b + a - i] = path[b + a - i], path[i]

    return path


def slide_mutation(path):
    a, b = random.sample(range(0, len(location) - 1), 2)
    if a > b:
        a, b = b, a
    t = path[a]
    for i in range(a + 1, b + 1):
        path[i - 1] = path[i]
    path[b] = t
    return path


def main():
    starttime = time.perf_counter()
    route = init_group(group_size)
    tempdis = []
    printnum=0
    for i in range(trying_times):
        parent = tournament(route, parent_size+100)
        route = nextgene(route, parent)
        # result = min([sumval(j) for j in route])
        result = 9999999
        tpath = []
        for j in range(len(route)):
            if sumval(route[j]) < result:
                result = sumval(route[j])
                tpath = route[j]
        tempdis.append(result)
        if (i+1) % 10 == 0:
            print('第', i+1, '次遗传，现在的距离为', result)
            print('现在的路线为', tpath)
            plt.ion()
            printfig(tpath, i+1,starttime)
    print('最终路线为：',route[-1])
    print('最终距离为',tempdis[-1])

    # 绘制进化次数-距离图
    plt.plot(tempdis)
    plt.title('Distance change with evolution times')
    plt.xlabel('evolution times')
    plt.ylabel('distance')

    plt.show()


if __name__ == '__main__':
    main()

'''
def chartchoice(old_group):
    funclist=groupfunc(old_group)
    lunpanlist=[]
    totalp=0
    for i in range(len(funclist)):
        totalp+=funclist[i]
    for i in range(len(funclist)):
        lunpanlist.append(funclist[i]/totalp)
    t = 0
    for i in range(len(lunpanlist)):
        t = t + lunpanlist[i]
        lunpanlist[i] = t
    parents=[]
    for i in range(len(old_group)):
        if np.random.random()<lunpanlist[i]:
            parents.append(old_group[i])
    print(len(parents))
    return parents
'''

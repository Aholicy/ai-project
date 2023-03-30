import numpy as np
import operator

import time



class Step:
    state = []
    f = 0
    g = 0
    h = 0
    id = []
    parent = None

    def __init__(self, m):
        self.state = m
        self.f = 0  # f(n)=g(n)+h(n)
        self.g = 0  # g(n)
        self.h = 0  # h(n)
        self.id = []
        self.parent = None
        for i in range(len(m)):
            for j in range(len(m[0])):
                self.id.append(m[i][j])

    def create(self, fn, gn, hn, tparent):
        self.f = fn
        self.g = gn
        self.h = hn
        self.parent = tparent


# 启发函数
def h(s, finalstep):
    a = 0
    for i in range(len(s.state)):
        for j in range(len(s.state[i])):
            if s.state[i][j] != finalstep.state[i][j]:
                a = a + 1
    return a


def mht(s, finalstep):
    dx = 0
    dy = 0
    tsum = 0
    for i in range(len(s.state)):
        for j in range(len(s.state[i])):
            if s.state[i][j] == 0:
                continue
            dy = abs(j - (s.state[i][j] - 1) % 4)
            dx = abs(i - (s.state[i][j] - 1) // 4)
            tsum = tsum + dx + dy
    return tsum


def list_sort(l):
    cmp = operator.attrgetter('f')
    l.sort(key=cmp)


# A*算法
def A_star(s, finalstep):
    tnum = 0
    global openlist
    openlist = [s]
    global closelist
    closelist = set()
    endnum = False
    while len(openlist) > 0 and endnum == False:  # 当open表不为空
        old = openlist[0]  # 取出open表的首节点
        if (old.state == finalstep.state).all():  # 判断是否与目标节点一致
            return old
        openlist.remove(old)  # 将old移出open表
        closelist.add(old)
        # 判断此时状态的空格位置
        for a in range(len(old.state)):
            for b in range(len(old.state[a])):
                if old.state[a][b] == 0:
                    break
            if old.state[a][b] == 0:
                break
        # 开始移动
        move = [[1, 0], [-1, 0], [0, 1], [0,-1]]
        for i in range(len(move)):
            na = a + move[i][0]
            nb = b + move[i][1]
            if na >= 4 or na < 0 or nb >= 4 or nb < 0:
                continue
            c = old.state.copy()
            c[a][b] = c[na][nb]
            c[na][nb] = 0
            flag = 1
            new = Step(c)
            tparent = old
            tgn = old.g + 1
            thn = mht(new, finalstep)
            tfn = tgn + thn
            new.create(tfn, tgn, thn, tparent)
            if (new.state == finalstep.state).all():
                endnum = True
                return new
                break
            for k in range(len(openlist)):
                if (new.state == openlist[k].state).all():
                    flag = 0
            if new in closelist:
                flag = 0
            if (flag == 1):
                openlist.append(new)  # 加入open表中
                list_sort(openlist)  # 排序
                print(tnum)
                tnum = tnum + 1


# 递归打印路径

def printpath(f):
    tf = []
    tnum = 0
    while f is not None:
        tf.append(f.state)
        f = f.parent
        tnum = tnum + 1
    for i in range(len(tf)):
        print("move", end=" ")
        print(i)
        print(tf.pop())


def main():
    starttime = time.perf_counter()
    sstate = [5, 1, 3, 4, 2, 7, 8, 12, 9, 6, 11, 15, 0, 13, 10, 14]
    fstate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
    index = 0
    smatrix = np.zeros((4, 4))
    fmatrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            smatrix[i][j] = sstate[index]
            fmatrix[i][j] = fstate[index]
            index = index + 1
    startstep = Step(smatrix)  # 初始状态
    finalstep = Step(fmatrix)  # 目标状态

    ans = A_star(startstep, finalstep)
    if ans:
        print("have answer as follow")
        printpath(ans)
    else:
        print("no answer")
    endtime = time.perf_counter()
    print('Running time: %s Seconds' % (endtime - starttime))
    input()

if __name__ == '__main__':
    main()
# [1, 2, 4, 8, 5, 7, 11, 10, 13, 15, 0, 3, 14, 6, 9, 12]
# [5, 1, 3, 4, 2, 7, 8, 12, 9, 6, 11, 15, 0, 13, 10, 14]
# [14, 10, 6, 0, 4, 9, 1, 8, 2, 3, 5, 11, 12, 13, 7, 5]
# [6, 10, 3, 15, 14, 8, 7, 11, 5, 1, 0, 2, 13, 12, 9, 4]

import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import matplotlib.pyplot as plt
# -*- coding: utf-8 -*-
vectorizer = TfidfVectorizer()
learn_rate = 0.003  # 学习率
train_times = 500


def sgn(x):
    return 1 / (np.exp(-x) + 1)


class Perceptron:  # data为输入的标准化过后的向量，target为标准
    def __init__(self, data, target):
        self.datanum = data.shape[0]  # 数据数目
        self.inputnum = data.shape[1]  # 输入层神经元个数
        self.hidnum = int((self.inputnum + 6) ** 0.5) + 1  # 隐藏层神经元个数
        self.outnum = 6  # 输出层神经元个数
        self.data = data  # 输入矩阵
        self.target = target  # 目标
        self.Target_vec = np.eye(6)  # one-shot形式的向量
        self.w1 = 2 * np.random.random((self.hidnum, self.inputnum)) - 1  # 输入层与隐藏层的权值
        self.w2 = 2 * np.random.random((self.outnum, self.hidnum)) - 1  # 隐藏层与输出层的权值
        self.b1 = 2 * np.random.random((self.hidnum, 1)) - 1  # 隐藏层的阈值
        self.b2 = 2 * np.random.random((self.outnum, 1)) - 1  # 输出层的阈值
        self.loss = 0
        self.allloss=[]

    def train(self, learn_rate):
        tdata = self.data
        tdata = tdata.toarray()
        # print(tdata)
        trying_times = 0
        while trying_times < train_times:
            trying_times += 1
            dif=0
            for i in range(self.datanum):
                x = tdata[i].reshape(-1, 1)
                tv = self.Target_vec[int(self.target[i]) - 1].reshape(-1, 1)  # 目标输出向量
                th = sgn(np.dot(self.w1, x) - self.b1)
                ty = sgn(np.dot(self.w2, th) - self.b2)
                t2 = (ty - tv) * (1 - ty) * ty
                t1 = np.dot(self.w2.T, t2) * (1 - th) * th
                self.w1 -= learn_rate * np.dot(t1, x.T)
                self.w2 -= learn_rate * np.dot(t2, th.T)
                self.b1 -= -learn_rate * t1
                self.b2 -= learn_rate * t2
                dif += np.square(ty-tv)
            self.loss=np.sum(dif)/(2*self.datanum)
            print(self.loss)
            self.allloss.append(self.loss)



    def test(self, data, target):
        rightnum = 0  # 正确数
        data_array = data.toarray()
        datanum = data.shape[0]

        for i in range(datanum):
            x = data_array[i].reshape(-1, 1)
            th = sgn(np.dot(self.w1, x) - self.b1)
            ty = sgn(np.dot(self.w2, th) - self.b2)
            emotion = np.argmax(ty)  # y中最大值的索引，代表当时的情感强度
            tt = int(target[i]) - 1
            if emotion == tt:
                rightnum += 1
                # print(out)
                # print(data[k])
            # rightrate=rightnum/data.shape[0]
        return rightnum / data.shape[0], rightnum


def datain(filepath):
    data = []  # 除了开头的两个数字以外的字符
    data_emotion = []  # 情感程度
    pattern = r'(\d+) (\d) ([a-zA-Z]+) (.+)'
    with open(filepath) as fp:
        fp.readline()  # 读取第一行
        for line in fp:
            match_res = re.match(pattern, line)
            data_emotion.append(match_res.group(2))
            data.append(match_res.group(4))
        # print(data)
        # print(data_emotion)
        return data, data_emotion


def main():
    #for i in range(20):
    #learn_rate = 0.01
    train, train_target = datain('./Classification/train.txt')  # 读取训练集
    test, test_target = datain('./Classification/test.txt')  # 读取测试集
    vectorizer.fit(train)
    standard_train = vectorizer.transform(train)  # 标准化
    # vectorizer.fit(test)
    standard_test = vectorizer.transform(test)
    model = Perceptron(standard_train, train_target)
    start_time = time.process_time()
    model.train(learn_rate)
    truerate, truenum = model.test(standard_test, test_target)
    end_time = time.process_time()
    print("正确个数为", truenum, "正确率为", truerate * 100, "%")
    print(f'耗时:{end_time - start_time}s')
    printfig(model.allloss)
    #print(model.allloss)
    #printfig(all_truerate, all_learnrate)
def printfig(allloss):
    plt.plot(allloss)
    plt.legend()
    #plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("Graph of loss changing by training time ")
    plt.show()


if __name__ == '__main__':
    main()

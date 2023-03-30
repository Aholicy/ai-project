import clause as cs
import codecs


def typein(clause_list):
    #   clause_list = []
    i = 0
   # f = codecs.open('data.txt', mode='r')  # 打开txt文件，以"utf-8’编码读取
    #line = f.readline()  # 以行的形式进行读取文件
    #tline = line
    tline =input()
    while tline != "":
    #while input()!="":

        if tline[0] == '(':
            tline = tline[1:-1]
            # 除去最外层括号
        clause_list.append([])
        t_str = ""
        #tline.replace(" ","")
        for j in range(len(tline)):
            if tline[j] == " ":
                continue
            t_str += tline[j]
            if tline[j] == ')':  # 分割谓词公式
                cloze = cs.Clause()
                cloze.createstr(t_str)
                clause_list[i].append(cloze)
                t_str = ""  # 一个谓词结束后清空字符串
        #tline = f.readline()
        tline=input()
        i = i + 1
    for i in range(len(clause_list)):
        for j in range(len(clause_list[i])):
            print(clause_list[i][j].element, end="")
        print("")

# def add_if(new_clause,clause_list):


def to_example(element, variable, example):
    for i in range(len(variable)):
        for j in range(1, len(element)):
            if element[j] == variable[i]:
                element[j] = example[i]


def ui():
    print('----')
    print("Enter your question")
    print("Please enter twice after you type in")
    print('----')


def main():
    # input
    ui()
    clause_list = []
    typein(clause_list)
    # end of input
    flag = True
    while flag:
        for i in range(len(clause_list)):
            if not flag:
                break
            if len(clause_list[i]) == 1:     # clause_list[i]只有一个谓词的子句，开始寻找后面的有相同的
                for j in range(0, len(clause_list)):     # clause_list[j]超过一个谓词的取或的子句
                    variable = []
                    example = []
                    position = -1
                    if not flag:
                        break
                    if i == j:
                        continue

                    # to_example 实例化
                    for k in range(len(clause_list[j])):     # 在子句clause_list[j]中找相同的谓词，
                        do_time=0
                        if clause_list[i][0].get_weici() == clause_list[j][k].get_weici() and clause_list[i][0].fei() != clause_list[j][k].fei():
                            position = k  # clause_list[j][k]找到的谓词
                            do_time = do_time+1  # 找到了那就可以结束这一次k的循环了
                            for t in range(len(clause_list[j][k].element)-1):
                                # 找到那个谓词有没有实例化，如果没有，进行实例化
                                if len(clause_list[j][k].element[t+1]) == 1 or len(clause_list[i][0].element[t+1]) == 1:
                                    variable.append(clause_list[j][k].element[t+1])
                                    example.append(clause_list[i][0].element[t+1])
                                elif clause_list[j][k].element[t+1] != clause_list[i][0].element[t+1]:
                                    position = -1  # 找到了谓词一样但是实例不一样
                                    break
                            if do_time!= 0 :
                                break

                    if position == -1:
                        continue  # not found
                    new_clause = []
                    for k in range(len(clause_list[j])):
                        if k != position:    # 把要合并的那一行的除了合并的谓词的其他谓词都提取出来到new_clause里，并且把里面的变量都实例化
                            cloze =cs.Clause()
                            cloze.createlist(clause_list[j][k].element)
                            to_example(cloze.element,variable,example)
                            new_clause.append(cloze)

                    if len(new_clause) == 1:
                        for s in range(len(clause_list)):
                            if len(clause_list[s]) == 1 and new_clause[0].element == clause_list[s][0].element:
                                position = -1
                                break
                    if position == -1:
                        continue
                    clause_list.append(new_clause)
                    print(len(clause_list), end=":   ")
                    th = chr(position+97)
                    if len(variable)==0:
                        print("R[", i+1, ",", j+1, th, "]", end="=")
                    else:
                        print("R[", i+1, ",", j+1, th, "]", end="(")
                    for m in range(len(variable)):
                        if m == len(variable)-1 :
                            print(variable[m], "=", example[m], ") = ", end="")
                        else:
                            print(variable[m], "=", example[m], end=",")
                    print("(",end="")
                    for n in range(len(clause_list[len(clause_list)-1])):
                        print(clause_list[len(clause_list)-1][n].element, end="")
                    print(")")
                    if len(new_clause) == 1 :
                        for n in range(len(clause_list)-1):  # clause_list[j]超过一个谓词的取或的子句
                            if len(clause_list[n]) == 1 and new_clause[0].get_weici() == clause_list[n][0].get_weici() \
                                    and new_clause[0].element[1:] == clause_list[n][0].element[1:] \
                                    and new_clause[0].fei() != clause_list[n][0].fei():
                                print(len(clause_list)+1, ": R[", n+1, ",", len(clause_list), "]() =  \t[]")
                                flag = False
                                break
    input();暂停

if __name__ == '__main__':
    main()

"""
On(aa,bb) 
On(bb,cc)
Green(aa)
¬Green(cc)
(¬On(x,y),¬Green(x),Green(y))

GradStudent(sue)
(¬GradStudent(x), Student(x))
(¬Student(x),HardWorker(x))
¬HardWorker(sue)

A(tony)
A(mike)
A(john)
L(tony, rain)
L(tony, snow)
(¬A(x), S(x), C(x))
(¬C(y), ¬L(y, rain))
(L(z, snow), ¬S(z))
(¬L(tony, u), ¬L(mike, u))
(L(tony, v), L(mike, v))
(¬A(w), ¬C(w), S(w))

"""

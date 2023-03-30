class Clause:
    #  单个谓词
    element = []

    def __init__(self):
        self.element = []

    def createstr(self, t_str):
        if t_str[0] == ',':
            t_str = t_str[1:]
        s = ""
        for i in range(len(t_str)):
            s += t_str[i]
            if t_str[i] == '(' or t_str[i] == ',' or t_str[i] == ')':
                self.element.append(s[0:-1])
                s = ""

    def createlist(self, list2):
        for i in range(len(list2)):
            self.element.append(list2[i])

    def fei(self):      # whether ~
        return self.element[0][0] == "¬"

    def get_weici(self):
        if self.fei():
            return self.element[0][1:]
        else:
            return self.element[0]


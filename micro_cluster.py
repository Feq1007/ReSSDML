# 微簇，注意需要修改为向量模式
import math
import torch

esp = 0.00005

class MicroCluster:
    def __init__(self, point, re=1, label=-1, radius=-1, lmda=0.01):
        # 簇基本信息
        self.n = 1
        self.nl = 1 if label == -1 else 0
        self.ls = point
        self.ss = torch.square(point)
        self.t = 0
        self.re = re  # 可靠性
        self.label = label if label!=-1 else -1
        self.radius = radius if radius!=-1 else -1  # 对于只有一个样本的mc，半径为默认值,在初始化时赋值

        # 其他计算参数
        self.espilon = 0.00005
        self.lmda = lmda
        self.radius_factor = 1

    # 增
    def insert(self, point, labeled=False):
        self.n += 1
        self.nl += 0 if labeled else 1
        self.ls = self.ls + point
        self.ss = self.ss + torch.square(point)
        self.t = 0

    def update_reliability(self, p, increase, exp=1):
        """
        该确定如何更新可靠性
        """
        if increase:  # 分类准确，或者针对无标签数据， 近邻：同增异减
            self.re += (1 - self.re) * math.pow(math.e, p - 1)
        else:
            self.re -= (1 - self.re) * math.pow(math.e, p - 1)

    def update(self, ):
        """
        通过时间更新可靠性
        """
        self.t += 1
        self.re = self.re * math.pow(math.e, - self.lmda * self.espilon * self.t)
        return self.re

    # 查
    def get_deviation(self):
        ls_mean = torch.sum(torch.square(self.ls / self.n))
        ss_mean = torch.sum(self.ss / self.n)
        variance = ss_mean - ls_mean
        radius = torch.sqrt(variance)
        return radius

    def get_center(self):
        return self.ls / self.n

    def update_time(self, p):
        global esp
        self.espilon = esp * p

    def get_radius(self):
        if self.n <= 1:
            return self.radius
        return max(self.radius,self.get_deviation() * self.radius_factor)

    def __str__(self):
        return f"n = {self.n}; nl = {self.nl}; label = {self.label}; ls = {self.ls.shape}; ss = {self.ss.shape}; t = {self.t}; re = {self.re}; ra = {self.get_radius()}\n"


if __name__ == '__main__':
    mc = MicroCluster(torch.Tensor([1, 2, 3]))
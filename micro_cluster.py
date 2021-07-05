# 微簇，注意需要修改为向量模式
import math
import torch


class MicroCluster:
    def __init__(self, lmda=0.1):
        # 簇基本信息
        self.n = 0
        self.nl = 0
        self.ls = 0
        self.ss = 0
        self.t = 0
        self.re = 0  # 可靠性
        self.label = -1
        self.radius = -1  # 对于只有一个样本的mc，半径为默认值,在初始化时赋值

        # 其他计算参数
        self.radius_factor = 1.8
        self.espilon = 0.00005
        self.lmda = lmda
        self.min_variance = math.pow(1, -5)

    # 增
    def insert(self, x, labeled=False):
        self.n += 1
        self.nl += 1 if labeled else 0
        self.ls = self.ls + x
        self.ss = self.ss + x * x
        self.t = 0

    # 改
    def merge(self, mc):
        """
        待修改可靠性合并
        """
        self.n += mc.n
        self.ls += mc.ls
        self.ss += mc.ss
        self.t = max(self.t, mc.t)
        self.re = (self.re + mc.re) / 2

    def update_reliability(self, p, increase, exp=1):
        """
        该确定如何更新可靠性
        """
        if increase:  # 分类准确，或者针对无标签数据， 近邻：同增异减
            self.re += (1 - self.re) * math.pow(p, exp)
        else:
            self.re -= (1 - self.re) * math.pow(p, exp / math.e)

    def update(self, ):
        """
        通过时间更新可靠性
        """
        self.t += 1
        self.re = self.re * math.pow(2, - self.lmda * self.espilon * self.t)
        return self.re

    # 查
    def get_deviation(self):
        ls_mean = self.ls / self.n
        ss_mean = self.ss / self.n
        variance = ss_mean - ls_mean ** 2
        radius = torch.sqrt(torch.sum(variance))
        return radius

    def get_center(self):
        return self.ls / self.n

    def get_radius(self):
        if self.n <= 1:
            return self.radius
        return self.get_deviation() * self.radius_factor

    def __str__(self):
        return f"n = {self.n}; nl = {self.nl}; label = {self.label}; ls = {self.ls.shape}; ss = {self.ss.shape}; t = {self.t}; re = {self.re}; ra = {self.get_radius()}\n"


if __name__ == '__main__':
    mc = MicroCluster()
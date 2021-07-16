import os
import numpy as np
import torch
import pandas as pd
from numpy import ndarray
from scipy.special import softmax
from HyperParameter import HyperParameter
from micro_cluster import MicroCluster
from metric_stream import MetricStream
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, roc_auc_score
from collections import Counter

class ReSSDML:
    def __init__(self, metric_stream, args):
        self.args = args
        self.ms = metric_stream

        # 初始化微簇
        self.mcs = []
        self.ulmcs = []
        self.avg_radius = 0
        self.initialization()

        # statistic info
        self.true_label = []
        self.predict_label = []
        self.labeled_num = 0

        self.re_list = []

    def print(self, topk):
        data = {'n':[],'nl':[],'label':[],'ls':[],'ss':[],'t':[],'re':[],'ra':[]}
        for mc in self.mcs:
            data['n'].append(mc.n)
            data['nl'].append(mc.nl)
            data['label'].append(mc.label)
            data['ls'].append(mc.ls)
            data['ss'].append(mc.ss)
            data['t'].append(mc.t)
            data['re'].append(mc.re)
            data['ra'].append(mc.get_radius().item())
        df_data = pd.DataFrame(data)
        print(df_data.describe())
        print(Counter(data['label']))

        data = {'n':[],'nl':[],'label':[],'ls':[],'ss':[],'t':[],'re':[],'ra':[]}
        for mc in self.unlabeled_mcs:
            data['n'].append(mc.n)
            data['nl'].append(mc.nl)
            data['label'].append(mc.label)
            data['ls'].append(mc.ls)
            data['ss'].append(mc.ss)
            data['t'].append(mc.t)
            data['re'].append(mc.re)
            data['ra'].append(mc.get_radius().item())
        df_data = pd.DataFrame(data)
        print(df_data.describe())
        print(Counter(data['label']))

        print(topk)
        if topk != '':
            print([self.mcs[idx].label for idx in topk[1]])
            print('re list : ',[self.mcs[idx].re for idx in topk[1]])
            print('re is ', self.compute_reliability(topk))
            print('ra is ', [self.mcs[idx].get_radius() for idx in topk[1]])

    def start(self):
        for i, (data, label) in enumerate(self.ms.stream_dataloader):
            data = data.cuda() if self.ms.cuda else data
            data = self.ms.embedding(data).cpu()
            label = label.item()

            known = True if label != -1 else False
            self.labeled_num += 1 if known else 0

            topk = self.topk(data, labeled=True)
            pred = self.classify(topk)
            self.true_label.append(self.ms.y_stream[i])
            self.predict_label.append(pred[0])

            if known:
                self.update_softmax(topk, label, self.ms.y_stream[i]) # self.ms.y_stream[i] 是 真实标签
                # self.update_manifold(topk, label, self.ms.y_stream[i]) # self.ms.y_stream[i] 是 真实标签

            self.add_point(data, pred, label, topk, known)
            self.decay_mcs()

            if (i+1) % self.args.logging_steps == 0:
                self.evaluation()
            if (i + 1) % self.args.logging_steps == 0:
                self.print(topk)
            if (i+1) % 1000 == 0:
                self.adjust()
                # pass
        self.evaluation()

    def adjust(self):
        # 是否需要解决类不平衡问题？通过控制存在时间来控制微簇数量
        ratio = Counter([mc.label for mc in self.mcs])
        labels = [k for k,v in ratio.items()]
        p = softmax(np.array([v for k,v in ratio.items()]))
        p = p / max(p)
        for mc in self.mcs:
            idx = labels.index(mc.label)
            mc.lmda = self.args.lmbda * p[idx]

    def initialization(self):
        data = torch.Tensor(self.ms.metric_dataset.data)
        label = self.ms.metric_dataset.label

        if self.ms.cuda:
            data = data.cuda()
        data = self.ms.embedding(data).cpu().numpy()

        # 通过聚类检查效果
        for l_ref in self.ms.labels:
            data_l = data[label==l_ref]
            if len(data_l) > self.args.init_k:
                kmeans = KMeans(n_clusters=self.args.init_k)
                kmeans.fit(data_l)
                kmeans_label = kmeans.labels_
                for l_temp in range(self.args.init_k):
                    mc = MicroCluster(lmda=self.args.lmbda)
                    mc.label = l_ref
                    mc.re = 1
                    mc.t = 0
                    for d in data_l[kmeans_label==l_temp]:
                        mc.insert(torch.Tensor(d), True)
                    self.mcs.append(mc)
            else:
                for d in data_l:
                    mc = MicroCluster(lmda=self.args.lmbda)
                    mc.label = l_ref
                    mc.re = 1
                    mc.t = 0
                    mc.insert(torch.Tensor(d), True)
                    self.mcs.append(mc)

        avg_radius = torch.mean(torch.Tensor([mc.get_radius() for mc in self.mcs if mc.n > 1]))
        self.avg_radius = avg_radius

        for mc in self.mcs:
            if mc.n <= 1:
                mc.radius = self.avg_radius
        print(self.avg_radius)
        self.print('')

    def compute_reliability(self, topk):
        # dic = {}
        # for i in range(len(topk[1])):
        #     if self.mcs[topk[1][i]].label not in dic.keys():
        #         dic[self.mcs[topk[1][i]].label] = 1
        #     dic[self.mcs[topk[1][i]].label] *= self.mcs[topk[1][i]].re / topk[0][i]
        # res = [v for k,v in dic.items()]
        # return max(softmax(res))
        p = softmax(np.array([self.mcs[idx].re / topk[0][i] for i,idx in enumerate(topk[1])]))
        labels = np.array([self.mcs[idx].label for idx in topk[1]])
        res = []
        for l in self.ms.labels:
            if l in labels:
                res.append(np.sum(p[labels==l]))
        return max(res)

    def topk(self, new_point,k=-1, labeled=True):
        k = self.args.K if k == -1 else k
        if labeled:
            centers = torch.stack([mc.get_center() for mc in self.mcs])
        else:
            centers = torch.stack([mc.get_center() for mc in self.unlabeled_mcs])
        dis = self.ms.distance.compute_mat(query_emb=centers, ref_emb=new_point).flatten()
        topk = torch.topk(dis, min(k, len(dis)), largest=False, sorted=True) # 如果个数小于k的话会报错
        return topk

    # 分类
    def classify(self, topk):
        mc_labels = [mc.label for mc in self.mcs]

        res = np.zeros(len(self.ms.labels))
        for idx in topk[1]:
            res[self.ms.labels.index(mc_labels[idx])] += 1

        pred_label = self.ms.labels[np.argmax(res)]
        for i,idx in enumerate(topk[1]):
            if self.mcs[idx].label == pred_label:
                pred_mcidx = idx
                dis = topk[0][i]
                break
        return pred_label, pred_mcidx, dis

    def update_softmax(self, topk, pred_label, true_label):
        # 通过距离更新可靠性
        correct = pred_label == true_label
        topk_dis = topk[0].flatten()
        pk = softmax(topk_dis)
        for i, idx in enumerate(topk[1]):
            if self.mcs[idx].label == true_label:
                self.mcs[idx].update_reliability(pk[i], True)
            else:
                self.mcs[idx].update_reliability(pk[i], False)
            if correct:
                self.mcs[idx].t /= 10
            else:
                self.mcs[idx].t *= 1.2

    # def update_manifold(self, topk, pred_label, true_label):
    #     correct = pred_label == true_label

    def insert_unlabeled(self, point, label, known=False):
        if len(self.unlabeled_mcs) <= 1:
            mc = MicroCluster()
            mc.label = label if known else -1
            mc.re = 1
            mc.t = 0
            mc.radius = self.avg_radius
            mc.insert(point.flatten(),known)
            self.unlabeled_mcs.append(mc)
        else:
            untopk = self.topk(point,labeled=False)
            if untopk[0][0] < self.unlabeled_mcs[untopk[1][0]].get_radius():
                if known:
                    mc = self.unlabeled_mcs[untopk[1][0]]
                    mc.insert(point.flatten(), labeled=True)
                    mc.label = label
                    mc.t = 0
                    mc.re = 1
                    self.unlabeled_mcs.pop(untopk[1][0])
                    self.mcs.append(mc)
                else:
                    self.unlabeled_mcs[untopk[1][0]].insert(point.flatten())
            else: # create
                if known:
                    self.create_mc(point, 1, None, label, known)
                else:
                    if len(self.unlabeled_mcs) > self.args.MAXUNC:
                        self.unlabeled_mcs.sort(key=lambda x: x.t, reverse=False)
                        self.unlabeled_mcs.pop(-10)
                    mc = MicroCluster()
                    mc.t = 0
                    mc.re = 1
                    mc.radius = self.avg_radius
                    mc.insert(point.flatten(),labeled=False)
                    self.unlabeled_mcs.append(mc)
        return True

    # 微簇管理
    def add_point(self, point, pred, label, topk, known=True):
        re = self.compute_reliability(topk)
        self.re_list.append(re)
        if re < 0.5 and not known:
            pass
        elif re < self.args.MINRE:
            self.insert_unlabeled(point, label, known)
        else:
            self.insert_labeled(point, re, pred, label, topk, known)

    def insert_labeled(self, point, re, pred, label, topk, known=True):
        if self._need_create_mc(pred, label, known):
            return self.create_mc(point, re, pred, label, known)
        else:
            self.mcs[pred[1]].insert(point.flatten())
            return pred[1]

    def _need_create_mc(self, pred, label, known=False):
        return (pred[2] > self.mcs[pred[1]].get_radius()) or (known and pred[0] != label)

    def create_mc(self, point, re, pred, label, known):
        if len(self.mcs) >= self.args.MAXC:
            self.drop()
        mc = MicroCluster(lmda=self.args.lmbda)
        mc.radius = self.avg_radius
        if known:
            mc.label = label
            mc.re = 1
            mc.insert(point.flatten(), labeled=True)
        else:
            mc.label = pred[0]
            mc.re = re
            mc.insert(point.flatten(), labeled=False)
        self.mcs.append(mc)
        return self.mcs[-1]

    def drop(self,):
        def key(elem):
            return elem.t
        self.mcs.sort(key=key,reverse=False) # 是否需要通过排序来解决，并且一次只删除一个，基本上每次都会删除
        self.mcs = self.mcs[:-5]

        # for i,mc in enumerate(self.mcs):
        #     if mc.re < self.args.MINRE:
        #         self.mcs.pop(i)

    def decay_mcs(self,):
        for i,mc in enumerate(self.mcs):
            re = mc.update()
            if re < self.args.MINRE:
                self.mcs.pop(i)
        for i,mc in enumerate(self.unlabeled_mcs):
            re = mc.update()
            if re < self.args.MINRE:
                self.unlabeled_mcs.pop(i)

    def evaluation(self):
        rep = classification_report(self.true_label, self.predict_label, digits=4)
        print(rep)


if __name__ == '__main__':
    args = HyperParameter()
    ms = MetricStream(args=args, model_type='normal')
    ressdml = ReSSDML(ms, args)
    ressdml.start()
    for mc in ressdml.mcs:
        print(mc)
    print("ture label    : ", ressdml.true_label)
    print("predict label : ", ressdml.predict_label)
    print("true number   : ", ressdml.labeled_num)
    print("reliable value: ", ressdml.re_list)
    print("accept new mc:", np.sum(np.array(ressdml.re_list)>args.MINRE))
    ressdml.evaluation()
    ressdml.print('')

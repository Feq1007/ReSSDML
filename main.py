import os
import numpy as np
import torch
import random
from scipy.special import softmax
from HyperParameter import HyperParameter
from micro_cluster import MicroCluster
from metric_stream import MetricStream
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, roc_auc_score

class ReSSDML:
    def __init__(self, metric_stream, args):
        self.args = args
        self.ms = metric_stream

        # 初始化微簇
        self.mcs = []
        self.avg_radius = 0
        self.initialization()

        # statistic info
        self.true_label = []
        self.predict_label = []
        self.labeled_num = 0

    def print(self):
        for mc in self.mcs:
            print(mc)
            if mc.label == -1:
                os._exit(0)


    def start(self):
        for i, (data, label) in enumerate(self.ms.stream_dataloader):
            data = data.cuda() if self.ms.cuda else data
            data = self.ms.embedding(data).cpu()
            label = label.item()

            known = True if label != -1 else False
            self.labeled_num += 1 if known else 0

            topk = self.topk(data)
            pred = self.classify(topk)
            self.true_label.append(self.ms.y_stream[i])
            self.predict_label.append(pred[0])

            if known:
                self.update_manifold(topk, self.ms.y_stream[i]) # self.ms.y_stream[i] 是 真实标签

            self.add_point(data, pred, label, topk, known)
            self.decay_mcs()

            if (i+1) % self.args.logging_steps == 0:
                self.evaluation()
        self.evaluation()

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
                    mc = MicroCluster(lmda=self.args.lmda)
                    mc.label = l_ref
                    mc.re = 1
                    mc.t = 0
                    for d in data_l[kmeans_label==l_temp]:
                        mc.insert(torch.Tensor(d), True)
                    self.mcs.append(mc)
            else:
                for d in data_l:
                    mc = MicroCluster(lmda=self.args.lmda)
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

        self.print()

    def compute_reliability(self, topk):
        res = np.zeros(len(self.ms.labels))
        for i in range(len(topk)):
            try:
                res[self.ms.labels.index(self.mcs[topk[1][i]].label)] /= topk[0][i]
            except:
                os._exit(0)
        return max(softmax(res))

    def topk(self, new_point):
        centers = torch.stack([mc.get_center() for mc in self.mcs])
        dis = self.ms.distance.compute_mat(query_emb=centers, ref_emb=new_point).flatten()
        topk = torch.topk(dis, min(self.args.K, len(dis)), largest=False, sorted=True) # 如果个数小于k的话会报错
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


    def update_manifold(self, topk, label):
        # 通过距离更新可靠性
        topk_dis = topk[0].flatten()
        pk = softmax(topk_dis)
        for i, idx in enumerate(topk[1]):
            if self.mcs[idx].label == label:
                self.mcs[idx].update_reliability(pk[i], True)
            else:
                self.mcs[idx].update_reliability(pk[i], False)
        return True

    # 微簇管理
    def add_point(self, point, pred, label, topk, known=True):
        if self._need_create_mc(pred, label, known):
            return self.create_mc(point, pred, label, topk, known)
        else:
            self.mcs[pred[1]].insert(point.flatten())
            return pred[1]

    def _need_create_mc(self, pred, label, known=False):
        return (pred[2] > self.mcs[pred[1]].get_radius()) or (known and pred[0] != label)


    def create_mc(self, point, pred, label, topk, known):
        if len(self.mcs) >= self.args.MAXC:
            self.drop()
            topk = self.topk(point)
        mc = MicroCluster(lmda=self.args.lmda)
        mc.radius = self.avg_radius
        if known:
            mc.label = label
            mc.re = 1
            mc.insert(point.flatten(), labeled=True)
        else:
            mc.label = pred[0]
            mc.re = self.compute_reliability(topk)
            mc.insert(point.flatten(), labeled=False)
        self.mcs.append(mc)
        return self.mcs[-1]

    def drop(self,):
        for i,mc in enumerate(self.mcs):
            if mc.re < self.args.MINRE:
                self.mcs.pop(i)

    def decay_mcs(self,):
        for i,mc in enumerate(self.mcs):
            re = mc.update()
            if re < self.args.MINRE:
                self.mcs.pop(i)

    def evaluation(self):
        rep = classification_report(self.true_label, self.predict_label)
        print(rep)


if __name__ == '__main__':
    args = HyperParameter()
    ms = MetricStream(args=args, model_type='normal')
    ressdml = ReSSDML(ms, args)
    ressdml.start()
    print("ture label: ", ressdml.true_label)
    print("predict label : ", ressdml.predict_label)
    print("true number : ", ressdml.labeled_num)
    ressdml.evaluation()
    ressdml.print()

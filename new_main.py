# 使用无标签微簇
import numpy as np
import torch
import pandas as pd
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

        self.create_num = 0
        self.re_list = []

    def print(self, topk):
        print('平均半径:', self.avg_radius)
        print('新建微簇个数：', self.create_num)
        data = {'n': [], 'nl': [], 'label': [], 'ls': [], 'ss': [], 't': [], 're': [], 'ra': []}
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

        data = {'n': [], 'nl': [], 'label': [], 'ls': [], 'ss': [], 't': [], 're': [], 'ra': []}
        for mc in self.ulmcs:
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
            print('re list : ', [self.mcs[idx].re for idx in topk[1]])
            print('re is ', self.compute_reliability(topk))
            print('ra is ', [self.mcs[idx].get_radius() for idx in topk[1]])

    def initialization(self):
        data = torch.Tensor(self.ms.metric_dataset.data)
        data = data.cuda() if self.ms.cuda else data
        data = self.ms.embedding(data).cpu().numpy()
        label = self.ms.metric_dataset.label

        # 通过聚类检查效果
        for l_ref in self.ms.labels:
            data_lref = data[label == l_ref]
            if len(data_lref) > self.args.init_k:
                kmeans = KMeans(n_clusters=self.args.init_k, )
                kmeans.fit(data_lref)
                kmeans_label = kmeans.labels_
                for l_temp in range(self.args.init_k):
                    data_ltemp = data_lref[kmeans_label == l_temp]
                    if len(data_ltemp) == 0:
                        continue
                    mc = MicroCluster(torch.Tensor(data_ltemp[0]), label=l_ref, lmda=self.args.lmbda)
                    for d in data_ltemp[1:]:
                        mc.insert(torch.Tensor(d), labeled=True)
                    self.mcs.append(mc)
            else:
                mc = MicroCluster(torch.Tensor(data_lref[0]), label=l_ref, lmda=self.args.lmbda)
                for d in data_lref[1:]:
                    mc.insert(torch.Tensor(d), labeled=True)
                self.mcs.append(mc)
        self.avg_radius = torch.max(torch.Tensor([mc.get_radius() for mc in self.mcs if mc.n > 1]))
        for mc in self.mcs:
            if mc.n <= 1:
                mc.radius = self.avg_radius

    def start(self):
        self.print('')
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
                self.update_softmax(topk, label, self.ms.y_stream[i])  # self.ms.y_stream[i] 是 真实标签
                # self.update_manifold(topk, label, self.ms.y_stream[i]) # self.ms.y_stream[i] 是 真实标签

            if (i + 1) % self.args.logging_steps == 0:
                self.evaluation()
            if (i + 1) % self.args.logging_steps == 0:
                self.print(topk)
            if (i + 1) % 1000 == 0:
                # self.adjust()
                pass

            self.add_point(data, pred, label, topk, known)
            self.decay_mcs()
        self.evaluation()

    def topk(self, new_point, k=-1, labeled=True):
        k = self.args.K if k == -1 else k
        if labeled:
            centers = torch.stack([mc.get_center() for mc in self.mcs])
        else:
            centers = torch.stack([mc.get_center() for mc in self.ulmcs])
        dis = self.ms.distance.compute_mat(query_emb=centers, ref_emb=new_point).flatten()
        topk = torch.topk(dis, min(k, len(dis)), largest=False, sorted=True)  # 如果个数小于k的话会报错
        return topk

    # 分类
    def classify(self, topk):
        mc_labels = [mc.label for mc in self.mcs]

        res = np.zeros(len(self.ms.labels))
        for idx in topk[1]:
            res[self.ms.labels.index(mc_labels[idx])] += 1

        pred_label = self.ms.labels[np.argmax(res)]
        for i, idx in enumerate(topk[1]):
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

    def adjust(self):
        # 是否需要解决类不平衡问题？通过控制存在时间来控制微簇数量
        ratio = Counter([mc.label for mc in self.mcs])
        labels = [k for k, v in ratio.items()]
        p = softmax(np.array([v for k, v in ratio.items()]))
        p = p / max(p)
        for mc in self.mcs:
            idx = labels.index(mc.label)
            mc.lmda = self.args.lmbda * p[idx]

    def compute_reliability(self, topk):
        # dic = {}
        # for i in range(len(topk[1])):
        #     if self.mcs[topk[1][i]].label not in dic.keys():
        #         dic[self.mcs[topk[1][i]].label] = 1
        #     dic[self.mcs[topk[1][i]].label] *= self.mcs[topk[1][i]].re / topk[0][i]
        # res = [v for k,v in dic.items()]
        # return max(softmax(res))
        p = softmax(np.array([self.mcs[idx].re / topk[0][i] for i, idx in enumerate(topk[1])]))
        labels = np.array([self.mcs[idx].label for idx in topk[1]])
        res = []
        for l in self.ms.labels:
            if l in labels:
                res.append(np.sum(p[labels == l]))
        return max(res)

    # 微簇管理
    def add_point(self, point, pred, label, topk, known=True):
        re = self.compute_reliability(topk)
        self.re_list.append(re)
        if re < self.args.minRE:
            self.insert_unlabeled(point, label, known)
        else:
            self.insert_labeled(point, re, pred, label, topk, known)

    def insert_unlabeled(self, point, label, known=False):
        if len(self.ulmcs) < 1:
            mc = MicroCluster(point.flatten(), radius=self.avg_radius, lmda=self.args.lmbda, )
            self.ulmcs.append(mc)
        else:
            untopk = self.topk(point, k=1, labeled=False)
            if untopk[0][0] < self.ulmcs[untopk[1][0]].get_radius():
                if known:  # 将无标签mc改为有标签mc
                    mc = self.ulmcs[untopk[1][0]]
                    mc.insert(point.flatten(), labeled=True)
                    mc.label = label
                    mc.t = 0
                    mc.re = 1
                    self.ulmcs.pop(untopk[1][0])
                    self.mcs.append(mc)
                else:
                    self.ulmcs[untopk[1][0]].insert(point.flatten())
            else:  # create
                if known:
                    self.create_mc(point, 1, None, label, known)
                else:
                    self.create_num += 1
                    if len(self.ulmcs) > self.args.maxUMC:
                        self.ulmcs.sort(key=lambda x: x.t, reverse=False)
                        self.ulmcs.pop(-10)  # 个数可调节
                    mc = MicroCluster(point.flatten(), radius=self.avg_radius)
                    self.ulmcs.append(mc)

    def insert_labeled(self, point, re, pred, label, topk, known=True):
        if self._need_create_mc(pred, label, known):
            return self.create_mc(point, re, pred, label, known)
        else:
            self.mcs[pred[1]].insert(point.flatten())
            return pred[1]

    def _need_create_mc(self, pred, label, known=False):
        return (pred[2] > self.mcs[pred[1]].get_radius()) or (known and pred[0] != label)

    def create_mc(self, point, re, pred, label, known):
        self.create_num += 1
        if len(self.mcs) >= self.args.maxMC:
            self.drop()
        mc = MicroCluster(point.flatten(), radius=self.avg_radius, lmda=self.args.lmbda)
        if known:
            mc.label = label
            mc.re = 1
        else:
            mc.label = pred[0]
            mc.re = re
        self.mcs.append(mc)
        return self.mcs[-1]

    def drop(self, ):
        def key(elem):
            return elem.t

        self.mcs.sort(key=key, reverse=False)  # 是否需要通过排序来解决，并且一次只删除一个，基本上每次都会删除
        self.mcs = self.mcs[:-5]

        for i, mc in enumerate(self.mcs):
            if mc.re < self.args.minRE:
                self.mcs.pop(i)

    def decay_mcs(self, ):
        for i, mc in enumerate(self.mcs):
            re = mc.update()
            if re < self.args.minRE:
                self.mcs.pop(i)
        for i, mc in enumerate(self.ulmcs):
            re = mc.update()
            if re < self.args.minRE:
                self.ulmcs.pop(i)

        # 重新计算avg_radius
        centers = [mc.get_radius() for mc in self.mcs if mc.n > 1]
        if len(centers) > 1:
            self.avg_radius = torch.max(torch.Tensor([mc.get_radius() for mc in self.mcs if mc.n > 1]))
            for mc in self.mcs:
                if mc.n <= 1:
                    mc.radius = self.avg_radius

    def evaluation(self):
        rep = classification_report(self.true_label, self.predict_label, digits=4)
        print(rep)

if __name__ == '__main__':
    args = HyperParameter()
    ms = MetricStream(args=args, model_type='normal')
    ms.evaluation()
    ressdml = ReSSDML(ms, args)
    ressdml.start()
    print("ture label    : ", ressdml.true_label)
    print("predict label : ", ressdml.predict_label)
    print("true number   : ", ressdml.labeled_num)
    print("reliable value: ", ressdml.re_list)
    print("accept new mc:", np.sum(np.array(ressdml.re_list) > args.minRE))
    ressdml.evaluation()
    ressdml.print('')
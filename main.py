# 使用无标签微触
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

PRINT = False


class ReSSDML:
    def __init__(self, metric_stream, args):
        self.args = args
        self.ms = metric_stream

        # 初始化微簇
        self.labeled_mcs = []
        self.unlabeled_mcs = []
        self.initialization()

        self.true_label = []
        self.predict_label = []
        self.labeled_num = 0

    def initialization(self):
        data = torch.Tensor(self.ms.metric_dataset.data)
        label = self.ms.metric_dataset.label

        data = data.cuda() if self.ms.cuda else data
        data = self.ms.embedding(data).cpu().numpy()

        # 通过聚类检查效果
        for l_ref in self.ms.labels:
            data_l = data[label == l_ref]
            kmeans = KMeans(n_clusters=self.args.init_k)
            kmeans.fit(data_l)
            kmeans_label = kmeans.labels_
            for l_temp in range(self.args.init_k):
                mc = MicroCluster()
                mc.label = l_ref
                mc.re = 1
                mc.t = 0
                for d in data_l[kmeans_label == l_temp]:
                    mc.insert(torch.Tensor(d), True)
                self.labeled_mcs.append(mc)

    def start(self):
        for i, (data, label) in enumerate(self.ms.stream_dataloader):
            known = random.uniform(0, 1) < self.args.semi_rate  # 半监督，随机选择部分为无标签数据
            self.labeled_num += 1 if known else 0

            data = torch.Tensor(data)
            data = data.cuda() if self.ms.cuda else data
            data = self.ms.embedding(data)
            data = data.cpu()

            dis_labeled = self.compute_distance(data, labeled=True)
            pred, topk_labeled = self.classify(dis_labeled, known)
            self.true_label.append(label.item())
            self.predict_label.append(pred[0])

            if known:
                self.update_manifold(dis_labeled, topk_labeled, label)

            #             if label != -1:
            #                 self.update_mainfold(dis_labeled, topk_labeled, label)

            self.add_point(data, pred, label, dis_labeled, known)
            self.decay_mcs()

            if i % self.args.eval_batch_size == 0:
                self.evaluation()

    def compute_reliability(self, dis):
        """
        距离不同label的最近的softmax概率
        """
        mc_label = np.array([(mc_idx, mc.label) for mc_idx, mc in enumerate(self.mcs)])
        dis_near = [min(dis[mc_label[:, 0][mc_label[:, 1] == label]]) for label in range(len(self.labels))]
        re = max(softmax(dis_near))
        return re

    def compute_distance(self, new_point, labeled=False):
        if labeled:
            centers = torch.stack([mc.get_center() for mc in self.labeled_mcs])
        else:
            if len(self.unlabeled_mcs) == 0:
                return np.finfo(np.float32).max
            centers = torch.stack([mc.get_center() for mc in self.unlabeled_mcs])
        return self.ms.distance.compute_mat(query_emb=centers, ref_emb=new_point).flatten()

    # 分类
    def classify(self, dis, known=False):
        """
        基于距离来预测
        """
        k = 1 if known else self.args.K

        mc_re = np.array([mc.re for mc in self.labeled_mcs])  # 可靠性
        redis = dis / mc_re
        topk = torch.topk(redis, k, largest=False, sorted=True)  # 距离最小的k个的值和索引

        mc_labels = np.array([mc.label for mc in self.labeled_mcs], dtype=np.int8)
        res = np.zeros(len(self.ms.labels))
        for idx in topk[1]:
            res[self.ms.labels.index(mc_labels[idx])] += 1

        pred_label = self.ms.labels[np.argmax(res)]
        for idx in topk[1]:
            if self.labeled_mcs[idx].label == pred_label:
                pred_mcidx = idx
                break
        return (pred_label, pred_mcidx), topk

    def update_manifold(self, dis, topk, label):
        # 通过强制topk的一致性来修改可靠性
        #         new_re = []
        #         ref = np.zeros(len(self.ms.labels))
        #         for idx in topk_idx:
        #             ref[self.ms.labels.index(self.mcs[idx].label)] = self.mcs[idx].re

        #         for idx in topk_idx:
        #             center = self.mcs[idx].get_center()
        #             dis = self.compute_distance(center)
        #             topk_value, topk_idx = torch.topk(dis,self.args.K+1,largest=False,sorted=True)
        # 通过距离更新可靠性
        topk_dis = np.array([dis[idx] for idx in topk])
        pk = softmax(topk_dis)
        for i, idx in enumerate(topk[1]):
            if self.labeled_mcs[idx].label == label:
                self.labeled_mcs[idx].update_reliability(pk[i], True)
            else:
                self.labeled_mcs[idx].update_reliability(pk[i], False)

    # 微簇管理
    def add_point(self, point, pred, label, dis_labeled, known=True):
        """
        如果点不再任何微簇里面，或者预测与真实不一样，则创建新微簇，
        否则直接添加
        """
        if len(self.unlabeled_mcs) > 0:
            dis_unlabeled = self.compute_distance(point, labeled=False)
            min_uidx = torch.argmin(dis_unlabeled)
            min_lidx = torch.argmin(dis_labeled)
            if (dis_unlabeled[min_uidx] < dis_labeled[min_lidx]):
                create_mc = self._need_create_mc(min_uidx, dis_unlabeled[min_uidx], pred[0], label, known,
                                                 near_labeled=False)
            else:
                create_mc = self._need_create_mc(min_lidx, dis_labeled[min_lidx], pred[0], label, known,
                                                 near_labeled=True)

            if create_mc:
                return self.create_mc(point, pred[0], label, known)
            else:
                self.labeled_mcs[pred[1]].insert(point)
                return pred[1]
        else:
            if self._need_create_mc():
                torch.argmin(torch.argmin(dis_labeled), min(dis_labeled), label, known, near_labeled=True)
                return self.create_mc(point, pred[0], label, known)
            else:
                self.labeled_mcs[pred[1]].insert(point)
            return pred[1]


def _need_create_mc(self, min_idx, min_dis, pred, label, known=False, near_labeled=False):
    if PRINT:
        print('_need_create_mc')
    if near_labeled:
        return (min_dis > self.labeled_mcs[min_idx].get_radius()) or (known and pred != label)
    else:
        return (min_dis > self.unlabeled_mcs[min_idx].get_radius()) or (known and pred != label)


def create_mc(self, point, pred, label, known):
    """
    为新数据创建微簇
    """
    if len(self.labeled_mcs) + len(self.unlabeled_mcs) >= self.args.MAXC:
        self.merge()
    mc = MicroCluster()
    mc.insert(point)
    mc.re = 1
    mc.t = 0
    if known:
        mc.label = label
        self.labeled_mcs.append(mc)
    else:
        mc.label = -1
        self.unlabeled_mcs.append(mc)


#def merge(self, ):
#    """
#    合并顺序为：两个最近无标签微簇，最近有标签和无标签微簇以及两个有标签微簇
#    """
#    if len(self.unlabeled_mcs) >= 2:
#        centers = [mc.get_center() for mc in self.unlabeled_mcs]
#        dis = self.ms.distance.compute_mat(query_emb=centers, ref_emb=centers)
#        print(dis)
#
#    for lid in range(len(self.labels)):
#        dis_min = sys.float_info.max
#        mcs = [mc for mc in self.mcs if mc.label_id == lid]
#        centers = [mc.get_center() for mc in mcs]
#        for i, c in enumerate(centers):
#            dis = torch.sum((torch.square(centers[i + 1:, 0] - c)), axis=1)
#            if dis_min > min(dis):
#                mc1 = i
#                mc2 = i + 1 + torch.argmin(dis)
#                dis_min = dmin
#        self.mcs[mc1].merge(self.mcs[mc2])


def decay_mcs(self, ):
    """
    更新微簇时间可靠性
    """
    for i, mc in enumerate(self.labeled_mcs):
        re = mc.update()
    #             if re < self.args.MINRE:
    #                 del(self.labeled_mcs[i])

    for i, mc in enumerate(self.unlabeled_mcs):
        re = mc.update()


#             if re < self.args.MINRE:
#                 del(self.unlabeled_mcs[i])

def evaluation(self):
    rep = classification_report(self.true_label, self.predict_label)
    print(rep)


if __name__ == '__main__':
    args = HyperParameter()
    ms = MetricStream(args=args, model_type='normal')
    ms.start_learn()
    ms.evaluation()
    ressdml = ReSSDML(ms, args)
    ressdml.start()
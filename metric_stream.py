import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from HyperParameter import HyperParameter
from net.shallow import Shallow
from pytorch_metric_learning.distances import BaseDistance, LpDistance
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning.miners import MultiSimilarityMiner, BatchEasyHardMiner
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import torch
from torch.utils.data import Dataset, DataLoader
import seaborn as sns

from sklearn.cluster import KMeans


class MetricDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class MetricStream:
    """
    完成包括度量函数的学习以及数据流dataloader
    """

    def __init__(self, args, model_type='normal'):
        """

        :rtype: object
        """
        self.args = args
        self.labels = []

        # read data
        x_init, x_stream, y_init, y_stream = self.__preprocess(args)
        in_dim = x_init.shape[1]

        # 初始化数据集
        self.metric_dataset = MetricDataset(x_init, y_init)
        self.metric_dataloader = DataLoader(self.metric_dataset, batch_size=args.metric_batch_size)

        # 验证数据集
        self.eval_dataset = MetricDataset(x_stream[:self.args.eval_rate], y_stream[:self.args.eval_rate])

        # 数据流数据集
        stream_dataset = MetricDataset(x_stream, y_stream)
        self.stream_dataloader = DataLoader(stream_dataset, batch_size=1)

        if model_type == 'normal':
            self.model = Shallow(in_dim, in_dim * 2, in_dim, int(in_dim*2/3))

        if self.args.device == 'cuda' and torch.cuda.is_available():
            self.cuda = True
            self.model.cuda()
        else:
            self.cuda = False
        self.distance = LpDistance(power=2)
        self.miner = BatchEasyHardMiner()
        self.loss_func = TripletMarginLoss(distance=self.distance)
        if self.args.metric_learning:
            self.start_learn()

    def __preprocess(self, args):
        data = pd.read_csv(os.path.join(args.data_dir, args.train_file), sep=args.sep, header=None, dtype='float32')
        X = data.iloc[:, :-1].values
        Y = data.iloc[:, -1].values
        Y = Y.astype(np.int8)

        ok = False
        if not ok:
            if args.concept_drift:  # 这里的采样是否考虑类别均衡
                x_init, x_stream, y_init, y_stream = train_test_split(X, Y, args.init_rate)
            else:
                x_init, x_stream, y_init, y_stream = train_test_split(X, Y, train_size=args.init_rate, shuffle=True)
            y_init_class = set(y_init.flatten().tolist())
            y_stream_class = set(y_stream.flatten().tolist())
            if y_init_class == y_stream_class:
                ok = True
        self.labels = list(y_init_class)
        return x_init, x_stream, y_init, y_stream

    def evaluation(self):
        """

        :rtype: object
        """
        with torch.no_grad():
            X = torch.Tensor(self.metric_dataset.data)
            if self.cuda:
                X = X.cuda()
            X = self.model(X)
            X = X.cpu().detach().numpy()
            y = self.metric_dataset.label

            # 通过聚类检查效果
            estimator = KMeans(n_clusters=len(self.labels))  # 构造聚类器
            estimator.fit(X)  # 聚类
            label_pred = estimator.labels_  # 获取聚类标签

            centroids = estimator.cluster_centers_  # 获取聚类中心
            inertia = estimator.inertia_  # 获取聚类准则的总和

            X_std = StandardScaler().fit_transform(X)
            tsne = TSNE(n_components=2)
            X_tsne = tsne.fit_transform(X_std)

            X_tsne_data = np.vstack((X_tsne.T, y)).T
            df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class'])
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2')
            plt.show()

    # 待优化更新
    def start_learn(self):
        for i in tqdm(range(self.args.metric_epochs)):
            optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.learning_rate)
            for i, (data, labels) in enumerate(self.metric_dataloader):
                optimizer.zero_grad()
                data = data.float()
                if self.cuda:
                    data = data.cuda()
                    labels = labels.cuda()
                embeddings = self.model(data)
                hard_pairs = self.miner(embeddings, labels)
                loss = self.loss_func(embeddings, labels, hard_pairs)
                loss.backward()
                optimizer.step()

    def embedding(self, x):
        if not self.args.metric_learning:
            return x
        with torch.no_grad():
            return self.model(x)


if __name__ == "__main__":
    args = HyperParameter()
    ms = MetricStream(args=args, model_type='normal')
    ms.evaluation()

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
    def __init__(self, args, model_type='normal'):
        self.args = args
        self.labels = []

        # read data
        x_init, x_stream, y_init, y_stream, y_semi_stream = self.__preprocess(args)
        in_dim = x_init.shape[1]
        self.y_stream = y_stream

        # 初始化数据集
        self.metric_dataset = MetricDataset(x_init, y_init)
        self.metric_dataloader = DataLoader(self.metric_dataset, batch_size=args.metric_batch_size)

        # 数据流数据集
        stream_dataset = MetricDataset(x_stream, y_semi_stream)
        self.stream_dataloader = DataLoader(stream_dataset, batch_size=1, shuffle=self.args.data_shuffle)

        # 距离函数：L2 Norm
        self.distance = LpDistance(power=2)

        self.cuda = False
        if self.args.metric_learning:
            self.miner = BatchEasyHardMiner()
            self.loss_func = TripletMarginLoss(distance=self.distance)
            if model_type == 'normal':
                self.model = Shallow(in_dim, in_dim * 2, in_dim, int(in_dim * 2 / 3))

            if self.args.device == 'cuda' and torch.cuda.is_available():
                self.cuda = True
                self.model.cuda()
            else:
                self.cuda = False
            self.start_learn()

    def __preprocess(self, args):
        data = pd.read_csv(os.path.join(args.data_dir, args.train_file), sep=args.sep, header=None, dtype='float32')
        X = data.iloc[:, :-1].values
        Y = data.iloc[:, -1].values
        Y = Y.astype(np.int8)

        self.labels = list(set(Y))  # 改为从文件读取
        init_rate = self.args.init_rate if self.args.init_rate > 1 else self.args.init_rate * len(Y)

        x_init, y_init = [], []
        for y_temp in self.labels:
            x = X[Y == y_temp]
            y = Y[Y == y_temp]
            x_init.extend(x[:int(init_rate / len(self.labels))])
            y_init.extend(y[:int(init_rate / len(self.labels))])
        x_init, y_init = np.stack(x_init), np.stack(y_init)
        x_stream, y_stream = X[self.args.init_rate:], Y[self.args.init_rate:]

        # 根据参数随机mask部分标签
        index = np.random.choice(np.arange(len(y_stream)), size=int(self.args.unlabeled_ratio * len(y_stream)),
                                 replace=False)
        y_semi_stream = y_stream.copy()
        y_semi_stream[index] = -1
        p = 0
        return x_init, x_stream, y_init, y_stream, y_semi_stream

    def evaluation(self):
        X = torch.Tensor(self.metric_dataset.data)
        X = X.cuda() if self.cuda else X
        X = self.embedding(X).cpu().numpy()
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

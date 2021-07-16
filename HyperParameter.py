import os

data_type = 'weather'
file = 'weather.csv'
data_dir = os.path.join('data', data_type)


class HyperParameter():
    def __init__(self, ):
        self.task = 'ReSSDML'
        self.train_file = file
        self.data_type = data_type
        self.data_dir = data_dir

        # 数据流相关
        self.sep = ','  # csv文件的seq类型
        self.init_rate = 1000  # 大于1则是个数，0-1则是比例
        self.metric_learning = False
        self.init_k = 30  # 初始化时每个类的微簇个数
        self.K = 5  # 有标签数据到来时更新周围的几个微簇
        self.maxMC = 1000  # 最多维护多少微簇
        self.maxUMC = 200
        self.unlabeled_ratio = 0.7
        self.minRE = 0.9
        self.lmbda = 1e-4  # 无标签数量大时，lmda也应该设置大一点
        self.logging_steps = 1000

        # 训练相关
        self.metric_epochs = 300
        self.metric_batch_size = 250
        self.eval_batch_size = 500
        self.learning_rate = 2e-5
        self.weight_decay = 0.0
        self.gradient_accumulation_steps = 0
        self.adam_epsilon = 1e-8
        self.dropout_rate = 0
        self.warmup_steps = -1
        self.max_grad_norm = 1
        self.device = 'cpu'
        self.cuda = '0'


if __name__ == '__main__':
    args = HyperParameter()
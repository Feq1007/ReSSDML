import os

file = 'kddcup.data.mm_std'
data_type = 'kddcup99'
data_dir = os.path.join('data', data_type, 'raw')
model_dir = os.path.join('data', data_type, 'model')
report_dir = os.path.join('data', data_type, 'report')

os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)


class HyperParameter():
    def __init__(self, ):
        self.task = 'ReSSDML'
        # 训练相关
        self.train_file = file
        self.data_type = data_type
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.report_dir = report_dir
        self.metric_epochs = 10
        self.metric_batch_size = 100
        self.eval_batch_size = 500
        self.learning_rate = 2e-5
        self.weight_decay = 0.0
        self.gradient_accumulation_steps = 0
        self.adam_epsilon = 1e-8
        self.dropout_rate = 0
        self.warmup_steps = -1
        self.max_grad_norm = 1
        self.max_steps = 10000
        self.logging_steps = 10000
        self.save_steps = -1
        self.device = 'cuda'
        self.cuda = '0'
        self.init_rate = 1000
        self.eval_rate = 1000
        self.sep = ','  # csv文件的seq类型
        self.data_shuffle = False

        # 数据流相关
        self.init_k = 10 # 初始化时每个类的微簇个数
        self.K = 5  # 有标签数据到来时更新周围的几个微簇
        self.MAXC = 1000  # 最多维护多少微簇
        self.MINDIS = 1  # 最短距离
        self.unlabeled_ratio = 0.9
        self.metric_learning = False
        self.MINRE = 0.9

if __name__ == '__main__':
    args = HyperParameter()
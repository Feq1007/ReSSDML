# 同一数据格式
import pandas as pd

# iris数据
path = 'data/iris/raw/iris.data'
def iris(path):
    data = pd.read_csv(path,sep=',',header=None)
    dic = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
    columns = data.columns
    data[columns[-1]] = data[columns[-1]].map(dic)
    data.to_csv(path.replace('.data','.dat'),header=None,index=False,sep=' ')
# iris(path)


# weather 数据集
path = 'data/weather/raw/weather.mat'
from scipy.io import loadmat
import numpy
annots = loadmat(path)
data = annots['weather']
numpy.savetxt(path.replace('.mat','.tsv'), data, delimiter=' ')
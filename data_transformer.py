# 同一数据格式
import pandas as pd

task = 'GSD'


# iris数据
def iris(path='data/iris/raw/iris.data'):
    data = pd.read_csv(path, sep=',', header=None)
    dic = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    columns = data.columns
    data[columns[-1]] = data[columns[-1]].map(dic)
    data.to_csv(path.replace('.data', '.dat'), header=None, index=False, sep=' ')


# iris(path)


# weather 数据集
def weather():
    path = 'data/weather/raw/weather.mat'
    from scipy.io import loadmat
    import numpy
    annots = loadmat(path)
    data = annots['weather']
    numpy.savetxt(path.replace('.mat', '.tsv'), data, delimiter=',')

    mm_std(path.replace('.mat','.tsv'),path.replace('.mat','.dsv'),sep=',')

# shuttle
def shuttle():
    path = 'data/shuttle/raw/shuttle_Norm.mat'
    from scipy.io import loadmat
    import numpy
    annots = loadmat(path)
    data = annots['shuttle_Norm']
    numpy.savetxt(path.replace('.mat', '.csv'), data, delimiter=',')


# spam 数据集
def spam():
    from scipy.io import arff
    import pandas as pd

    filepath = 'data/spam/spam_data.arff'

    data = arff.loadarff(filepath)
    df = pd.DataFrame(data[0])
    # print(df.describe())
    # print(df.head())
    # # 样本
    # 对标签进行处理
    sample = df.values[:, :-1]
    label = df.values[:, -1]
    cla = []
    dic = {
        b'spam': 0,
        b'legitimate': 1
    }
    for i in label:
        cla.append(dic[i])
    data = np.concatenate([sample, np.array(cla).reshape([-1, 1])], axis=1).astype(np.int8)
    # np.save('data/spam/spam.array', data)
    df = pd.DataFrame(data)
    df.to_csv('data/spam/spam.csv', index=None, columns=None, header=None)

# UCI
def uci():
    with open('data/UCI HAR Dataset/train/X_train.txt','r') as f:
        length = []
        for line in f.readlines():
            l = line.strip().split(' ')
            if len(l) not in length:
                length.append(len(l))
        print(length)

    X = pd.read_csv('data/UCI HAR Dataset/train/X_train.txt', header=None, sep=' ')
    print(X)
    X = X.values
    print(X)
    y = pd.read_csv('data/UCI HAR Dataset/train/y_train.txt', header=None, sep=',').values
    df_mm = MinMaxScaler().fit_transform(X)
    df_mm = pd.DataFrame(df_mm * 100)
    print('df_mm:\n', df_mm.describe())
    newdata = np.concatenate([df_mm, np.array(y, dtype=np.int8).reshape([-1, 1])], axis=1)
    df = pd.DataFrame(newdata)
    df.to_csv(des, index=None, columns=None, header=None)

# GSD
def gsd():
    # for i in range(1,11):
    #     file = f'data/GSD/batch{i}.dat'
    #     with open(file.replace('.dat','.csv'),'w') as fout:
    #         with open(file) as f:
    #             for line in f.readlines():
    #                 line = line.strip().split(' ')
    #                 t = []
    #                 for j in line[1:]:
    #                     t.append(j.split(':')[1]+',')
    #                 s = ''.join(t)
    #                 fout.write(s+line[0]+'\n')
    # for i in range(1,11):
    #     file = f'data/GSD/batch{i}.csv'
    #     mm_std(file,file.replace('.csv','.tsv'),',')
    with open('data/GSD/gsd.csv', 'w') as fout:
        lines = []
        for i in range(1,11):
            with open(f'data/GSD/batch{i}.tsv') as f:
                lines.extend(f.readlines())
        fout.writelines(lines)
# kddcup99
import numpy as np
import pandas as pd
import csv
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split

global label_list
label_list = []

src_file = 'data/kddcup99/raw/kddcup.data_10_percent_corrected'
des_file = 'data/kddcup99/raw/kddcup.data_10_percent_corrected.csv'


def preprocess(src, des):
    data_file = open(des, 'w', newline='')
    with open(src) as data_source:
        csv_reader = csv.reader(data_source)
        csv_writer = csv.writer(data_file)
        count = 0
        for i, row in enumerate(csv_reader):
            temp_line = np.array(row)
            temp_line[1] = handleProtocol(row)
            temp_line[2] = handleService(row)
            temp_line[3] = handleFlag(row)
            temp_line[41] = handleLabel(row)
            csv_writer.writerow(temp_line)
            count += 1
            if count % 100000 == 0:
                print(temp_line)
    data_file.close()


def handleProtocol(input):
    """
    定义将源文件行中3种协议类型转换成数字标识的函数
    """
    protocol_list = ['tcp', 'udp', 'icmp']
    if input[1] in protocol_list:
        return protocol_list.index(input[1])


def handleService(input):
    """
    定义将源文件行中70种网络服务类型转换成数字标识的函数
    """
    service_list = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u',
                    'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest',
                    'hostnames',
                    'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell',
                    'ldap',
                    'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp',
                    'nntp',
                    'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje',
                    'shell',
                    'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time',
                    'urh_i', 'urp_i',
                    'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
    if input[2] in service_list:
        return service_list.index(input[2])


def handleFlag(input):
    """
    定义将源文件行中11种网络连接状态转换成数字标识的函数
    """
    flag_list = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    if input[3] in flag_list:
        return flag_list.index(input[3])


def handleLabel(input):
    """
    定义将源文件行中攻击类型转换成数字标识的函数(训练集中共出现了22个攻击类型，而剩下的17种只在测试集中出现)
    """
    global label_list
    if not input[41] in label_list:
        label_list.append(input[41])
    return label_list.index(input[41])


def kddcup99():
    preprocess(src_file, des_file)

    df = pd.read_csv(des_file, header=None)

    df_mm = MinMaxScaler().fit_transform(df)
    df_mm = pd.DataFrame(df_mm)
    print('df_mm:\n', df_mm.describe())

    df_mm_std = StandardScaler().fit_transform(df_mm.iloc[:, :-1])
    df_mm_std = pd.DataFrame(df_mm_std)
    print('df_mm_std:\n', df_mm_std.describe())

    newdata = np.concatenate([df_mm_std, np.array(df.iloc[:, -1], dtype=np.int8).reshape([-1, 1])], axis=1)
    df = pd.DataFrame(newdata)
    df.to_csv('kddcup.data.mm_std', index=None, columns=None, header=None)


def mm_std(path, des, sep):
    df = pd.read_csv(path, header=None, sep=sep)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    df_mm = MinMaxScaler().fit_transform(X)
    df_mm = pd.DataFrame(df_mm * 100)
    print('df_mm:\n', df_mm.describe())
    newdata = np.concatenate([df_mm, np.array(y, dtype=np.int8).reshape([-1, 1])], axis=1)

    # df_mm_std = StandardScaler().fit_transform(df_mm)
    # df_mm_std = pd.DataFrame(df_mm_std)
    # print('df_mm_std:\n', df_mm_std.describe())

    # df_mm_std = Normalizer().fit_transform(df_mm)
    # df_mm_std = pd.DataFrame(df_mm_std)
    # print('df_mm_std:\n', df_mm_std.describe())
    # newdata = np.concatenate([df_mm_std, np.array(y, dtype=np.int8).reshape([-1, 1])], axis=1)

    df = pd.DataFrame(newdata)
    df.to_csv(des, index=None, columns=None, header=None)


# sensor task
def sensor():
    src = 'data/sensor/raw/sensor.txt'
    des = 'data/sensor/raw/sensor.csv'
    mm_std(src, des, ' ')

if task == 'kddcup99':
    # kddcup99()
    src = 'data/kddcup99/raw/kddcup.data_10_percent_corrected.csv'
    des = 'data/kddcup99/raw/kddcup.data.tsv'
    mm_std(src,des,',')
elif task == 'iris':
    iris()
elif task == 'weather':
    weather()
elif task == 'spam':
    spam()
elif task == 'sensor':
    sensor()
elif task == 'shuttle':
    shuttle()
elif task == 'UCI':
    uci()
elif task == 'GSD':
    gsd()
# 同一数据格式
import pandas as pd

task = 'sensor'


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
    numpy.savetxt(path.replace('.mat', '.tsv'), data, delimiter=' ')


# spam 数据集
def spam():
    from scipy.io import arff
    import pandas as pd

    filepath = 'spam/spam_data.arff'
    data = arff.loadarff(filepath)
    df = pd.DataFrame(data[0])
    # 样本
    sample = df.values[:, 0:len(df.values[0]) - 1]
    # 对标签进行处理
    # [b'1' b'-1' ...]bytes类型
    label = df.values[:, -1]  # 要处理的标签
    cla = []  # 处理后的标签
    for i in label:
        test = int(i)
        cla.append(test)
    cla


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
    df_mm = pd.DataFrame(df_mm)
    print('df_mm:\n', df_mm.describe())

    df_mm_std = StandardScaler().fit_transform(df_mm)
    df_mm_std = pd.DataFrame(df_mm_std)
    print('df_mm_std:\n', df_mm_std.describe())

    newdata = np.concatenate([df_mm_std, np.array(y, dtype=np.int8).reshape([-1, 1])], axis=1)
    df = pd.DataFrame(newdata)
    df.to_csv(des, index=None, columns=None, header=None)


# sensor task
def sensor():
    src = 'data/sensorless_drive/raw/Sensorless_drive_diagnosis.txt'
    des = 'data/sensorless_drive/raw/Sensorless_drive_diagnosis.csv'
    mm_std(src, des, ' ')

if task == 'kddcup99':
    kddcup99()
elif task == 'iris':
    iris()
elif task == 'weather':
    weather()
elif task == 'spam':
    spam()
elif task == 'sensor':
    sensor()
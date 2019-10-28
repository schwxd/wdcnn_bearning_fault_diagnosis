from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同


def prepro(d_path, framesize, trainnumber, testnumber, normal=True):
    """对数据进行预处理,返回train_X, train_Y, test_X, test_Y样本.

    :param d_path: 源数据地址
    :param length: 信号长度，默认2个信号周期，864
    :param number: 每种信号个数,总共10类,默认每个类别1000个数据
    :param normal: 是否标准化.True,Fales.默认True
    :return: Train_X, Train_Y, Test_X, Test_Y

    """
    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)

    def capture(original_path):
        """读取mat文件，返回字典

        :param original_path: 读取路径
        :return: 数据字典
        """
        files = {}
        filenames = os.listdir(original_path)
        for i in filenames:
            # 文件路径
            file_path = os.path.join(d_path, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if args.defe in key:
                    files[i] = file[key].ravel()
        return files

    def slice_enc(data, trainnumber, testnumber, framesize):
        """将数据切分为前面多少比例，后面多少比例.

        :param data: 单挑数据
        :return: 切分好的数据
        """
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]
            all_lenght = len(slice_data)
            Train_sample = []
            Test_Sample = []
            # 抓取测试数据，不用数据增强
            for h in range(testnumber):
                sample = slice_data[h*framesize:(h+1)*framesize]
                Test_Sample.append(sample)

            # 抓取训练数据，使用数据增强
            for j in range(trainnumber):
                random_start = np.random.randint(low=testnumber * framesize, high=all_lenght-framesize)
                sample = slice_data[random_start:random_start + framesize]
                Train_sample.append(sample)

            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        labels_dict = {}
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            labels_dict[i] = label
            label += 1
        print('labels_dict {}'.format(labels_dict))
        return X, Y

    # one-hot编码
    def one_hot(Train_Y, Test_Y):
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        Test_Y = np.array(Test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        Test_Y = Encoder.transform(Test_Y).toarray()
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

    def scalar_stand(Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    # 将数据手动乱序
    def shuffle_data(features, labels):
        rand_index = np.arange(len(features))
        np.random.shuffle(rand_index)
        features_shuffled = features[rand_index]
        labels_shuffled = labels[rand_index]
        return features_shuffled, labels_shuffled

    # 从所有.mat文件中读取出数据的字典
    data = capture(original_path=d_path)
    print(data.keys())
    # 将数据切分为训练集、测试集
    train, test = slice_enc(data, trainnumber, testnumber, framesize)
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)
    # 为训练集Y/测试集One-hot标签
    # Train_Y, Test_Y = one_hot(Train_Y, Test_Y)
    # 训练数据/测试数据 是否标准化.
    if normal:
        Train_X, Test_X = scalar_stand(Train_X, Test_X)
    else:
        # 需要做一个数据转换，转换成np格式.
        Train_X = np.asarray(Train_X)
        Test_X = np.asarray(Test_X)

    Train_Y = np.asarray(Train_Y)
    Test_Y = np.asarray(Test_Y)
    Train_X, Train_Y = shuffle_data(Train_X, Train_Y)
    Test_X, Test_Y = shuffle_data(Test_X, Test_Y)
    return Train_X, Train_Y, Test_X, Test_Y

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=False, default='data/deonly', help='where the data folder is')
parser.add_argument('--framesize', type=int, required=False, default=2048, help='frame size')
parser.add_argument('--trainnumber', type=int, required=False, default=660, help='how many samples each class for training dataset')
parser.add_argument('--testnumber', type=int, required=False, default=25, help='how many samples each class for test dataset')
parser.add_argument('--defe', required=False, default='DE', help='DE or FE')
parser.add_argument('--outputpath', required=False, default='output/deonly', help='where the output folder is')
args = parser.parse_args()

if __name__ == "__main__":
    subdirs = os.listdir(args.dataroot)
    print('args: {}'.format(args))
    print('subdirs: {}'.format(subdirs))
    for subdir in subdirs:
        res_dir = '{}'.format(os.path.join(args.outputpath, subdir))
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        datapath = os.path.join(args.dataroot, subdir)
        train_X, train_Y, test_X, test_Y = prepro(d_path=datapath,
                                                    framesize=args.framesize,
                                                    trainnumber=args.trainnumber,
                                                    testnumber=args.testnumber,
                                                    normal=True) 
             
        np.save("{}/data_features_train.npy".format(res_dir), train_X)
        np.save("{}/data_labels_train.npy".format(res_dir), train_Y)
        np.save("{}/data_features_test.npy".format(res_dir), test_X)
        np.save("{}/data_labels_test.npy".format(res_dir), test_Y)
        print(" train features {}, test features {}".format(train_X.shape, test_X.shape))
        print(" train labels {}, test labels {}".format(train_Y.shape, test_Y.shape))
        print('save result to {}'.format(res_dir))
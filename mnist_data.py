
# 将mnist官网下载下来的数据转为pkl，代码参考百度经验’mnist数据集怎么用‘，
# 地址为https://jingyan.baidu.com/article/414eccf6a45c9b6b431f0a2a.html
import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)

    images_path = os.path.join(path, '%s-images.idx3-ubyte'% kind)

    with open(labels_path, 'rb') as lbpath:

        magic, n = struct.unpack('>II',lbpath.read(8))

        labels = np.fromfile(lbpath,dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

train_images, train_labels = load_mnist('dataset',kind='train')
train_sample = len(train_labels)
train_y = np.zeros((train_sample,10))
for i in range(train_sample):
    train_y[i,train_labels[i]]=1

test_images, test_labels = load_mnist('dataset',kind='t10k')
test_sample = len(test_labels)
test_y = np.zeros((test_sample,10))
for i in range(test_sample):
    test_y[i,test_labels[i]]=1

import pickle as pkl
data_path = "dataset/data.npz"
savemodel = open(data_path, 'wb')
data =  {'train_x': train_images, 'train_y':train_y, 'test_x':test_images, 'test_y':test_y}
pkl.dump(data, savemodel)
savemodel.close()
# NeuralNetwork

1. 使用numpy库来实现BP神经网络，并使用mnist手写数字识别来验证。

2. 使用pytorch库的Tensor来实现BP神经网络，并使用mnist手写数字识别来验证。

# 数据集

dataset文件夹是mnist官网下载的数据，data.npz是使用mnist_data.py将数据转换成能更容易读取的npz格式.

而test.csv和train.csv是kaggle的新手赛Digit Recognizer的数据集，与官网的数据集在测试集和训练集的数量上不同之外，应该没多大差别，groundtruth.csv是test.csv的真实标签（由该比赛的大佬的给出，他的得分是1）。


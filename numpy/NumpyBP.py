import numpy as np
import pickle as pkl

class BP:  # BP神经网络
    def __init__(self, sl, test_x, test_y, train_X=None, train_Y=None, learn_rate=1):
        self.train_X = train_X  # (样本数n_sample, 特征数sl[0])
        self.train_Y = train_Y  # (样本数n_sample, 特征数sl[0])
        self.n_sample = train_Y.shape[0]  # 样本个数
        self.learn_rate = learn_rate  # 学习率
        self.L = len(sl) - 1  # 层数
        self.sl = sl  # 每层的神经元个数
        self.lambd = 0.001

        '''
        self.w = [0]  # 权重
        self.b = [0]  # 偏置项
        for l in range(1,self.L):
            self.w.append(np.random.randn(self.sl[l] ,sl[l +1])*0.01)  # 初始化第l层神经元的权重w
            self.b.append(np.zeros((1 ,sl[ l +1])))  # 初始化第l层偏置项b
        self.w.append(0)
        self.b.append(0)
        '''
        cnn_model_save_path = "NumpyBP.pkl"
        savemodel = open(cnn_model_save_path, 'rb')
        trained_model = pkl.load(savemodel)
        # get weight
        self.w = trained_model['w']
        self.b = trained_model['b']

        self.a = 0
        self.z = 0

        self.test_x = test_x
        self.test_y = test_y
        self.accuracy = 0.21
        self.cost = 0.42

    def forward(self, x, w=None, b=None):  # 前向传导
        z = [0, 0]  # 求和值
        a = [0, x]  # 激活值
        if w == None:
            '''是否定义w和b:
            自定义代表着用已保存的模型来预测结果
            未定义代表着训练以及在训练同时测试，看训练效果，此时用已有的数据成员即可
            '''
            w = self.w
            b = self.b
        for l in range(2, self.L + 1):
            z.append(a[l - 1].dot(w[l - 1]) + b[l - 1])
            a.append(self.relu(z[l]))
        self.z = z
        self.a = a
        return a[-1]

    def costfunction(self):  # 损失函数

        costsum = 0

        costsum += 1 / 2 * (np.linalg.norm(self.a[-1] - self.train_Y)) ** 2

        return costsum / self.n_sample

    def backward_propagation(self):  # 反向传播，用的是批量梯度下降法
        sigma = list(np.zeros((self.L + 1, 1)))  # 一个列表，每个元素是一层神经元的残差
        for l in range(self.L, 1, -1):
            if l == self.L:  ##所有样本在最后一层的残差,大小(n_sample,sl[l-1])
                sigma[l] = np.multiply((self.a[l] - self.train_Y), self.relu_derivative(self.z[l]))
            else:  # 在其他层的残差，并更新偏置项b和权重w

                sigma[l] = np.zeros((self.n_sample, self.sl[l]))  # 这一层所有样本的残差计算，大小(n_sample,sl[l-1])
                for j in range(self.n_sample):  # 每个样本的残差
                    b1 = np.mat(self.relu_derivative(self.z[l])[j,])
                    b2 = np.mat(sigma[l + 1][j, :])
                    d = np.multiply(self.w[l].dot(b2.T), b1.T)
                    sigma[l][j,] = d.T

                # 更新偏置项b
                self.b[l] -= self.learn_rate * np.sum(sigma[l + 1], axis=0, keepdims=True) / self.n_sample

                dw = np.zeros((self.n_sample, self.sl[l], self.sl[l + 1]))  # 每个样本的dw（关于权重w的偏导）
                for j in range(self.n_sample):
                    b1 = np.mat(self.a[l][j,]).T
                    b2 = np.mat(sigma[l + 1][j, :])
                    dw[j, :, :] = np.dot(b1, b2)
                # 每个样本的dw的和
                sum_dw = np.sum(dw, axis=0, keepdims=True).reshape((self.sl[l], self.sl[l + 1]))
                # 更新权重w
                self.w[l] -= self.learn_rate * (sum_dw / self.n_sample + self.lambd * self.w[l])

    def relu(self, z):  # relu激活函数
        return np.maximum(z, 0.0)

    def relu_derivative(self, z):  # relu激活函数的导数
        s = 1. * (z > 0)
        return s

    def train(self, epoch):  # 训练
        for i in range(0, epoch):  # 进行迭代
            self.forward(self.train_X)
            self.backward_propagation()
            cost = self.costfunction()

            print(i, cost)

            accuracy = self.acc()
            print('acc:', accuracy)

            if accuracy > self.accuracy or cost < self.cost:
                self.accuracy = accuracy
                self.cost = cost
                trained_model = {'w': self.w, 'b': self.b}
                model_save_path = "NumpyBP.pkl"
                savemodel = open(model_save_path, 'wb')
                pkl.dump(trained_model, savemodel)
                savemodel.close()

            if cost < 0.01:
                break

    def predict(self, x, w, b):  # 预测结果
        predict = self.forward(x, w=w, b=b)
        predict = np.argmax(predict, axis=1)
        return predict

    def acc(self):
        pre = self.forward(self.test_x)
        pre = np.argmax(pre, axis=1)
        accuracy = np.mean(pre == self.test_y)
        return accuracy


if __name__ == '__main__':
    # read data
    data = np.load('../dataset/data.npz', allow_pickle=True)
    train_x = data['train_x'][:10000, ] / 255.0
    train_y = data['train_y'][:10000, ]

    test_x = data['train_x'][50000:, ] / 255.0
    test_y = data['train_y'][50000:, ]
    test_y = np.argmax(test_y, axis=1)

    print('数据处理完毕')
    sl = [0, train_x.shape[1], 300, 100, 10]
    nn = BP(sl, test_x, test_y, train_X=train_x, train_Y=train_y)

    # nn.train(1000)  # 训练

    cnn_model_save_path = "NumpyBP.pkl"
    savemodel = open(cnn_model_save_path, 'rb')
    trained_model = pkl.load(savemodel)

    # get weight
    w = trained_model['w']
    b = trained_model['b']
    # predict
    pred_x = data['test_x']/255.0
    groundtruth = np.argmax(data['test_y'], axis=1)
    pre = nn.forward(pred_x)
    predict = np.argmax(pre, axis=1)
    # evluate
    accuracy = np.mean(predict == groundtruth)
    print('accuracy:', accuracy)

    import pandas as pd
    pre_x = np.array(pd.read_csv("../test.csv"))/255.0
    num = np.array([i + 1 for i in range(pre_x.shape[0])])
    pre = nn.predict(pre_x, w=w, b=b)
    # pd_data = pd.DataFrame(np.column_stack((num, pre)), columns=['ImageId', 'Label'])
    # pd_data.to_csv('submission.csv', index=None)

    Groundtruth = pd.read_csv("../groundtruth.csv")
    Groundtruth = np.array(Groundtruth.loc[:, "Label"]).T
    accuracy = np.mean(pre == Groundtruth)
    print('accuracy:', accuracy)

# -- coding: utf-8 -- #
# 基础BP神经网络
import numpy as np
import sys
from sklearn import preprocessing


# Sigmoid启动函数
def _sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# Relu启动函数
def _relu(x, derive=False):
    if derive:
        np.where(x > 0, 1, 0)
        return x
    return np.maximum(x, 0.0)


# 三层神经网络
class Layer3:

    def __init__(self, active, layers, iteration):
        self.active = active
        self.layers = layers
        self.iteration_times = iteration
        self.accuracy = 0.0

    def train(self, x_train, y_train, x_predict):
        # 获取激活函数
        active = getattr(sys.modules["algorithm.ann"], "_" + self.active)
        # 获取隐藏层数
        hide_node_num = self.layers[0]
        # 设置随机种子，保证初始的权重不变
        np.random.seed(1)
        # 归一化处理
        x_normalize = preprocessing.MinMaxScaler()
        y_normalize = preprocessing.MinMaxScaler()
        x = x_normalize.fit_transform(x_train)
        y = y_normalize.fit_transform(y_train)
        sample = x_normalize.fit_transform(x_predict)
        # 获取输入层与输出层节点数
        input_node_num = len(x[0])
        output_node_num = len(y[0])

        # 输入层为0，下一层为1，三层神经网络，输出层即2层
        # 随机生产初始权重，np.random.random()产[0,1]以内的随机数，乘2变为[0,2]，-1变为[-1,1]。
        # [输入数，隐藏层节点数]
        weight0 = 2 * np.random.random((input_node_num, hide_node_num)) - 1
        # [隐藏层节点数，1]
        bias0 = np.random.random((hide_node_num, 1)) - 1
        # [隐藏层节点数，输出节点数]
        weight1 = 2 * np.random.random((hide_node_num, output_node_num)) - 1
        # [输出节点数，1]
        bias1 = np.random.random((output_node_num, 1)) - 1

        layer2_loss = 0

        for iteration in range(self.iteration_times):
            # 将输入赋给输入层[样本数，输入数]
            layer0 = x
            # 隐藏层1：获取预测值，并使用启动函数归一化[样本数，输入数]dot[输入数，隐藏层节点数]+[隐藏层节点数，1].T=[样本数，隐藏层节点数]
            layer1 = active(np.dot(layer0, weight0) + bias0.T)
            # 隐藏层2：获取预测值，并使用启动函数归一化[样本数，隐藏层节点数]dot[隐藏层节点数，输出节点数]+[输出节点数，1].T=[样本数，输出节点数]
            layer2 = active(np.dot(layer1, weight1) + bias1.T)
            # 计算每次实例更新的损失[样本数，输出节点数]
            layer2_loss = y - layer2
            # 输出层每次实例更新的误差[样本数，输出节点数]*[样本数，输出节点数]=[样本数，输出节点数]
            layer2_delta = layer2_loss * active(layer2, True)
            # 计算每次实例更新的损失[样本数，输出节点数]*[输出节点数，隐藏层节点数]=[样本数，隐藏层节点数]
            layer1_loss = np.dot(layer2_delta, weight1.T)
            # 隐藏层每次实例更新的误差[样本数，隐藏层节点数]]*[样本数，隐藏层节点数]=[样本数，隐藏层节点数]
            layer1_delta = layer1_loss * active(layer1, True)

            # 输出层权重更新值计算[隐藏层节点数，样本数]dot[样本数，输出节点数]=[隐藏层节点数，输出节点数]
            weight1 += np.dot(layer1.T, layer2_delta)
            # 输出层偏倚更新值计算[输出节点数，1]+[输出节点数，1]=[输出节点数，1]
            bias1 += np.array([layer2_delta.sum(axis=0)]).T
            # 隐藏层权重更新值计算[输入数，样本数]dot[样本数，隐藏层节点数]=[输入数，隐藏层节点数]
            weight0 += np.dot(layer0.T, layer1_delta)
            # 隐藏层偏倚更新值计算[隐藏层节点数，1]]+[隐藏层节点数，1]]=[隐藏层节点数，1]]
            bias0 += np.array([layer1_delta.sum(axis=0)]).T

        # 预测模型精度
        self.accuracy = '%.2f%%' % ((1 - abs(layer2_loss.sum() / y.sum())) * 100)

        # 获得预测结果
        sample_layer1 = active(np.dot(sample, weight0) + bias0.T)
        sample_layer2 = active(np.dot(sample_layer1, weight1) + bias1.T)
        return y_normalize.inverse_transform(sample_layer2)


# 四层神经网络
class Layer4:

    def __init__(self, active, layers, iteration):
        self.accuracy = 0.0
        self._active = active
        self._layers = layers
        self._iteration_times = iteration

    def train(self, x_train, y_train, x_predict):
        # 获取激活函数
        active = getattr(sys.modules["algorithm.ann"], "_" + self._active)
        # 获取隐藏层数
        hide_node_num_1 = self._layers[0]
        hide_node_num_2 = self._layers[1]
        # 设置随机种子，保证初始的权重不变
        np.random.seed(1)
        # 归一化处理
        x_normalize = preprocessing.MinMaxScaler()
        y_normalize = preprocessing.MinMaxScaler()
        x = x_normalize.fit_transform(x_train)
        y = y_normalize.fit_transform(y_train)
        sample = x_normalize.fit_transform(x_predict)
        # 获取输入层与输出层节点数
        input_node_num = len(x[0])
        output_node_num = len(y[0])

        # 输入层为0，下一层为1，四层神经网络，输出层即3层
        # 随机生产初始权重，np.random.random()产[0,1]以内的随机数，乘2变为[0,2]，-1变为[-1,1]。
        # [输入数，隐藏层1节点数]
        weight0 = 2 * np.random.random((input_node_num, hide_node_num_1)) - 1
        # [隐藏层1节点数，1]
        bias0 = np.random.random((hide_node_num_1, 1)) - 1
        # [隐藏层1节点数，隐藏层2节点数]
        weight1 = 2 * np.random.random((hide_node_num_1, hide_node_num_2)) - 1
        # [隐藏层2节点数，1]
        bias1 = np.random.random((hide_node_num_2, 1)) - 1
        # [隐藏层2节点数，输出节点数]
        weight2 = 2 * np.random.random((hide_node_num_2, output_node_num)) - 1
        # [输出节点数，1]
        bias2 = np.random.random((output_node_num, 1)) - 1

        layer3_loss = 0

        for iteration in range(self._iteration_times):
            # 将输入赋给输入层[样本数，输入数]
            layer0 = x
            # 隐藏层1：获取预测值，并使用启动函数归一化[样本数，输入数]dot[输入数，隐藏层1节点数]+[隐藏层1节点数，1].T=[样本数，隐藏层1节点数]
            layer1 = active(np.dot(layer0, weight0) + bias0.T)
            # 隐藏层2：获取预测值，并使用启动函数归一化[样本数，隐藏层1节点数]dot[隐藏层1节点数，隐藏层2节点数]+[隐藏层2节点数，1].T=[样本数，隐藏层2节点数]
            layer2 = active(np.dot(layer1, weight1) + bias1.T)
            # 隐藏层3：获取预测值，并使用启动函数归一化[样本数，隐藏层2节点数]dot[隐藏层2节点数，输出节点数]+[输出节点数，1].T=[样本数，输出节点数]
            layer3 = active(np.dot(layer2, weight2) + bias2.T)
            # 计算每次实例更新的损失[样本数，输出节点数]
            layer3_loss = y - layer3
            # 输出层每次实例更新的误差[样本数，输出节点数]*[样本数，输出节点数]=[样本数，输出节点数]
            layer3_delta = layer3_loss * active(layer3, True)

            # 计算每次实例更新的损失[样本数，输出节点数]dot[输出节点数，隐藏层2节点数]=[样本数，隐藏层2节点数]
            layer2_loss = np.dot(layer3_delta, weight2.T)
            # 隐藏层每次实例更新的误差[样本数，隐藏层2节点数]*[样本数，隐藏层2节点数]=[样本数，隐藏层2节点数]
            layer2_delta = layer2_loss * active(layer2, True)

            # 计算每次实例更新的损失[样本数，隐藏层2节点数]dot[隐藏层2节点数，隐藏层1节点数]=[样本数，隐藏层1节点数]
            layer1_loss = np.dot(layer2_delta, weight1.T)
            # 隐藏层每次实例更新的误差[样本数，隐藏层1节点数]]*[样本数，隐藏层1节点数]=[样本数，隐藏层1节点数]
            layer1_delta = layer1_loss * active(layer1, True)

            # 输出层权重更新值计算[隐藏层2节点数，样本数]dot[样本数，输出节点数]=[隐藏层2节点数，输出节点数]
            weight2 += np.dot(layer2.T, layer3_delta)
            # 输出层偏倚更新值计算[输出节点数，1]+[输出节点数，1]=[输出节点数，1]
            bias2 += np.array([layer3_delta.sum(axis=0)]).T
            # 隐藏层2权重更新值计算[隐藏层1节点数，样本数]dot[样本数，隐藏层2节点数]=[隐藏层1节点数，隐藏层2节点数]
            weight1 += np.dot(layer1.T, layer2_delta)
            # 隐藏层2偏倚更新值计算[隐藏层2节点数，1]+[隐藏层2节点数，1]=[隐藏层2节点数，1]
            bias1 += np.array([layer2_delta.sum(axis=0)]).T
            # 隐藏层1权重更新值计算[输入数，样本数]dot[样本数，隐藏层1节点数]=[输入数，隐藏层1节点数]
            weight0 += np.dot(layer0.T, layer1_delta)
            # 隐藏层1偏倚更新值计算[隐藏层1节点数，1]]+[隐藏层1节点数，1]]=[隐藏层1节点数，1]]
            bias0 += np.array([layer1_delta.sum(axis=0)]).T

        # 预测模型精度
        # self.accuracy = '%.2f%%' % ((1 - abs(layer3_loss.sum() / y.sum())) * 100)
        # 获得预测结果
        sample_layer1 = active(np.dot(sample, weight0) + bias0.T)
        sample_layer2 = active(np.dot(sample_layer1, weight1) + bias1.T)
        sample_layer3 = active(np.dot(sample_layer2, weight2) + bias2.T)
        return y_normalize.inverse_transform(sample_layer3).astype(int)


# 四层神经网络
class Layer5:

    def __init__(self, active, layers, iteration):
        self.accuracy = 0.0
        self._active = active
        self._layers = layers
        self._iteration_times = iteration

    def train(self, x_train, y_train, x_predict):
        # 获取激活函数
        active = getattr(sys.modules["algorithm.ann"], "_" + self._active)
        # 获取隐藏层数
        hide_node_num_1 = self._layers[0]
        hide_node_num_2 = self._layers[1]
        hide_node_num_3 = self._layers[2]
        # 设置随机种子，保证初始的权重不变
        np.random.seed(1)
        # 归一化处理
        x_normalize = preprocessing.MinMaxScaler()
        y_normalize = preprocessing.MinMaxScaler()
        x = x_normalize.fit_transform(x_train)
        y = y_normalize.fit_transform(y_train)
        sample = x_normalize.fit_transform(x_predict)
        # 获取输入层与输出层节点数
        input_node_num = len(x[0])
        output_node_num = len(y[0])

        # 输入层为0，下一层为1，四层神经网络，输出层即3层
        # 随机生产初始权重，np.random.random()产[0,1]以内的随机数，乘2变为[0,2]，-1变为[-1,1]。
        # [输入数，隐藏层1节点数]
        weight0 = 2 * np.random.random((input_node_num, hide_node_num_1)) - 1
        # [隐藏层1节点数，1]
        bias0 = np.random.random((hide_node_num_1, 1)) - 1
        # [隐藏层1节点数，隐藏层2节点数]
        weight1 = 2 * np.random.random((hide_node_num_1, hide_node_num_2)) - 1
        # [隐藏层2节点数，1]
        bias1 = np.random.random((hide_node_num_2, 1)) - 1
        # [隐藏层2节点数，隐藏层3节点数]
        weight2 = 2 * np.random.random((hide_node_num_2, hide_node_num_3)) - 1
        # [隐藏层3节点数，1]
        bias2 = np.random.random((hide_node_num_3, 1)) - 1
        # [隐藏层3节点数，输出节点数]
        weight3 = 2 * np.random.random((hide_node_num_3, output_node_num)) - 1
        # [输出节点数，1]
        bias3 = np.random.random((output_node_num, 1)) - 1

        layer4_loss = 0

        for iteration in range(self._iteration_times):
            # 将输入赋给输入层[样本数，输入数]
            layer0 = x
            # 隐藏层1：获取预测值，并使用启动函数归一化[样本数，输入数]dot[输入数，隐藏层1节点数]+[隐藏层1节点数，1].T=[样本数，隐藏层1节点数]
            layer1 = active(np.dot(layer0, weight0) + bias0.T)
            # 隐藏层2：获取预测值，并使用启动函数归一化[样本数，隐藏层1节点数]dot[隐藏层1节点数，隐藏层2节点数]+[隐藏层2节点数，1].T=[样本数，隐藏层2节点数]
            layer2 = active(np.dot(layer1, weight1) + bias1.T)
            # 隐藏层3：获取预测值，并使用启动函数归一化[样本数，隐藏层2节点数]dot[隐藏层2节点数，隐藏层3节点数]+[隐藏层3节点数，1].T=[样本数，隐藏层3节点数]
            layer3 = active(np.dot(layer2, weight2) + bias2.T)
            # 隐藏层4：获取预测值，并使用启动函数归一化[样本数，隐藏层3节点数]dot[隐藏层3节点数，输出节点数]+[输出节点数，1].T=[样本数，输出节点数]
            layer4 = active(np.dot(layer3, weight3) + bias3.T)

            # 计算每次实例更新的损失[样本数，输出节点数]
            layer4_loss = y - layer4
            # 输出层每次实例更新的误差[样本数，输出节点数]*[样本数，输出节点数]=[样本数，输出节点数]
            layer4_delta = layer4_loss * active(layer4, True)

            # 计算每次实例更新的损失[样本数，输出节点数]dot[输出节点数，隐藏层2节点数]=[样本数，隐藏层3节点数]
            layer3_loss = np.dot(layer4_delta, weight3.T)
            # 隐藏层每次实例更新的误差[样本数，隐藏层3节点数]*[样本数，隐藏层3节点数]=[样本数，隐藏层3节点数]
            layer3_delta = layer3_loss * active(layer3, True)

            # 计算每次实例更新的损失[样本数，隐藏层3节点数]dot[隐藏层3节点数，隐藏层2节点数]=[样本数，隐藏层2节点数]
            layer2_loss = np.dot(layer3_delta, weight2.T)
            # 隐藏层每次实例更新的误差[样本数，隐藏层2节点数]*[样本数，隐藏层2节点数]=[样本数，隐藏层2节点数]
            layer2_delta = layer2_loss * active(layer2, True)

            # 计算每次实例更新的损失[样本数，隐藏层2节点数]dot[隐藏层2节点数，隐藏层1节点数]=[样本数，隐藏层1节点数]
            layer1_loss = np.dot(layer2_delta, weight1.T)
            # 隐藏层每次实例更新的误差[样本数，隐藏层1节点数]]*[样本数，隐藏层1节点数]=[样本数，隐藏层1节点数]
            layer1_delta = layer1_loss * active(layer1, True)

            # 输出层权重更新值计算[隐藏层3节点数，样本数]dot[样本数，输出节点数]=[隐藏层3节点数，输出节点数]
            weight3 += np.dot(layer3.T, layer4_delta)
            # 输出层偏倚更新值计算[输出节点数，1]+[输出节点数，1]=[输出节点数，1]
            bias3 += np.array([layer4_delta.sum(axis=0)]).T
            # 输出层权重更新值计算[隐藏层2节点数，样本数]dot[样本数，输出节点数]=[隐藏层2节点数，输出节点数]
            weight2 += np.dot(layer2.T, layer3_delta)
            # 输出层偏倚更新值计算[输出节点数，1]+[输出节点数，1]=[输出节点数，1]
            bias2 += np.array([layer3_delta.sum(axis=0)]).T
            # 隐藏层2权重更新值计算[隐藏层1节点数，样本数]dot[样本数，隐藏层2节点数]=[隐藏层1节点数，隐藏层2节点数]
            weight1 += np.dot(layer1.T, layer2_delta)
            # 隐藏层2偏倚更新值计算[隐藏层2节点数，1]+[隐藏层2节点数，1]=[隐藏层2节点数，1]
            bias1 += np.array([layer2_delta.sum(axis=0)]).T
            # 隐藏层1权重更新值计算[输入数，样本数]dot[样本数，隐藏层1节点数]=[输入数，隐藏层1节点数]
            weight0 += np.dot(layer0.T, layer1_delta)
            # 隐藏层1偏倚更新值计算[隐藏层1节点数，1]]+[隐藏层1节点数，1]]=[隐藏层1节点数，1]]
            bias0 += np.array([layer1_delta.sum(axis=0)]).T

        # 预测模型精度
        # self.accuracy = '%.2f%%' % ((1 - abs(layer3_loss.sum() / y.sum())) * 100)
        # 获得预测结果
        sample_layer1 = active(np.dot(sample, weight0) + bias0.T)
        sample_layer2 = active(np.dot(sample_layer1, weight1) + bias1.T)
        sample_layer3 = active(np.dot(sample_layer2, weight2) + bias2.T)
        sample_layer4 = active(np.dot(sample_layer3, weight3) + bias3.T)
        return y_normalize.inverse_transform(sample_layer4).astype(int)

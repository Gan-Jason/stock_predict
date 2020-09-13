import properties
import sys
import datetime
from algorithm import ann


class PredictService:
    def __init__(self, data_service):
        # 读取参数配置文件 todo print: properties reading succeed
        # 参数赋值 todo print: properties setting succeed
        self.active = 'sigmoid'  # 设置数据读取延迟
        self.layers = [3, 3]  # 数据读取时长
        self.iteration = 100  # 学习样本天数
        self._algorithm = ann.Layer4(self.active, self.layers, self.iteration)  # 算法获取
        self._data_service = data_service

    def predict(self, x_history, y_history, x_real):
        result = self._algorithm.train(x_history, y_history, x_real)
        return result

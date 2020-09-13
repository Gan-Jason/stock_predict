import sys
from util import properties
import datetime
import numpy as np
from chinese_calendar import is_workday


class DataService:
    def __init__(self, data_dao):
        props = properties.Properties(sys.path[0].split('stocks_predict.py')[0] + "/data.properties")
        self.x_length = int(props.get("x_length"))
        self.y_length = int(props.get("y_length"))
        self.sample_num = int(props.get("sample_num"))
        self.stock_list = props.get("stock_list").split(' ')
        self.data_dao = data_dao

    def get_x_history(self, cur_time):
        target_time = self.create_x_history_times(cur_time, self.sample_num, self.y_length)
        results = []
        data_flag = False
        for a_time in target_time:
            period_result = []
            period_flag = False
            start = self.create_start_time(a_time, self.x_length)
            for stock_num in self.stock_list:
                result = self.data_dao.get_data(start, a_time, stock_num)
                result = np.array(result)
                if not period_flag:
                    period_result = result
                    period_flag = True
                    continue
                period_result = np.hstack(period_result, result)
            period_result = np.array(period_result.flatten()).T
            # print('start:',start,'end:',a_time,'result_length:',period_result.size)
            if not data_flag:
                results = period_result
                data_flag = True
                continue
            results = np.row_stack((results, period_result))
        return results

    def get_y_history(self, cur_time):
        target_time = self.create_y_history_time(cur_time, self.sample_num)
        results = np.array([])
        data_flag = False
        for a_time in target_time:
            period_result = np.array([])
            period_flag = False
            start = self.create_start_time(a_time, self.y_length)
            for stock_num in self.stock_list:
                result = self.data_dao.get_data(start, a_time, stock_num)
                result = np.array(result)
                if not period_flag:
                    period_result = result
                    period_flag = True
                    continue
                period_result = np.hstack(period_result, result)
            period_result = np.array(period_result.flatten()).T
            if not data_flag:
                results = period_result
                data_flag = True
                continue
            results = np.row_stack((results, period_result))
        return results

    def get_x_real(self, cur_time):
        results = np.array([])
        start = self.create_start_time(cur_time, self.x_length)
        data_flag = False
        for stock_num in self.stock_list:
            result = self.data_dao.get_data(start, cur_time, stock_num)
            result = np.array(result)
            if not data_flag:
                results = result
                data_flag = True
                continue
            results = np.hstack(results, result)
        results = np.array(results.flatten()).T
        results = results.reshape(1, -1)
        return results

    @staticmethod
    def create_x_history_times(cur_time, sample_num, length):
        history_times = []
        i = length
        while True:
            target_time = cur_time - datetime.timedelta(i)
            if target_time.weekday() < 5 and is_workday(target_time):  # 工作日股市才开盘
                history_times.append(target_time)
            i += 1
            if len(history_times) == sample_num:
                break
        return history_times

    @staticmethod
    def create_y_history_time(cur_time, sample_num):
        history_times = []
        i = 0
        while True:
            target_time = cur_time - datetime.timedelta(i)
            if target_time.weekday() < 5 and is_workday(target_time):  # 工作日股市才开盘
                history_times.append(target_time)
            i += 1
            if len(history_times) == sample_num:
                break
        return history_times

    @staticmethod
    def create_start_time(end_time, length):
        i = 1
        start_time = end_time
        while i < length:
            start_time -= datetime.timedelta(1)
            if start_time.weekday() < 5 and is_workday(start_time):
                i += 1

        return start_time

import pandas_datareader.data as data
import datetime
import numpy as np


class StocksDao(object):
    def __init__(self):
        pass

    def get_data(self, start, end, stock_num):
        stock = data.DataReader(stock_num, "yahoo", start, end)
        results = np.array(stock.values).T
        return results

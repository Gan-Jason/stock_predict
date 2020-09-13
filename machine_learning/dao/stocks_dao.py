import pandas_datareader.data as data
import datetime
import numpy as np


class StocksDao(object):
    def __init__(self):
        pass

    @staticmethod
    def get_data(start, end, stock_num):
        stock = data.DataReader(stock_num, "yahoo", start, end)
        return stock.values


class DataDaoFactory:
    @staticmethod
    def create():
        return StocksDao()

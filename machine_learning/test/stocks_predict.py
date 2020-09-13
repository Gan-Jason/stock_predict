import pandas_datareader.data as web
import datetime
from chinese_calendar import is_workday
from chinese_calendar import is_holiday
import matplotlib.pyplot as plt
import copy
from sklearn import svm
import numpy as np
import tqdm


def stocks_api():
    start = datetime.datetime(2020, 6, 26)
    end = datetime.datetime(2020, 8, 27)
    stock = web.DataReader('603866.SS', "yahoo", start, end)
    print(stock.head(5))
    print(stock.tail(5))
    print(stock.index)
    print(stock.columns)
    change = stock.Close.diff()
    stock['Change'] = change
    stock['pct_change'] = 100.0 * (stock['Change']) / stock['Close'].shift(1)
    plt.plot(stock['Close'], 'r', color='#FF0000')
    plt.show()

    # paramsDim = 100
    # predictNum = 100
    # y = []
    # x = []
    # params = [0] * paramsDim
    # for j in range(1, paramsDim + 1):
    #     params[j - 1] = stock.iloc[j]['pct_change']
    #
    # for i in range(paramsDim + 1, stock.shape[0] - predictNum):
    #     pct_change = stock.iloc[i]['pct_change']
    #     y.append(pct_change)
    #     x.append(copy.deepcopy(params))
    #     params[:-1] = params[1:]
    #     params[-1] = pct_change
    #
    # clf = svm.SVR()
    # clf.fit(x, y)
    #
    #
    # result = []
    # for i in tqdm.tqdm(range(predictNum)):
    #     for j in range(paramsDim):
    #         params[j] = stock.iloc[stock.shape[0] - paramsDim + j - predictNum + 1 + i]['pct_change']
    #         print(params[-1])
    #         result.append(clf.predict([params]))
    #
    # plt.plot(stock.index[-predictNum:], result[::100], linewidth='1', label='test', color='#00FF00', linestyle=':', marker='|')
    # plt.plot(stock[-predictNum:]['pct_change'], 'r', color='#FF0000')
    # plt.show()


def numpy_test():
    ary1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ary2 = np.array([[10, 11, 12], [13, 14, 15], [16, 26, 37]])
    # result = np.hstack((ary1,ary2))
    # result = np.vstack((ary1,ary2))
    print(ary1.flatten())


if __name__ == '__main__':
    stocks_api()

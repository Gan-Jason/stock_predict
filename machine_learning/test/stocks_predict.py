import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import copy
from sklearn import svm
import numpy as np
import tqdm

start = datetime.datetime(2020, 6, 12)
end = datetime.datetime.today()
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

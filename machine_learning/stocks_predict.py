from service import data_service
from service import predict_service
from dao import stocks_dao
import datetime
from chinese_calendar import is_workday
import logging
import sys

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(sys.path[0].split('stocks_predict.py')[0] + "/stocks_predict.log")
file_handler.setLevel(logging.DEBUG)
data = data_service.DataService(stocks_dao.DataDaoFactory.create())
predict = predict_service.PredictService(data)


def job(cur_time):
    logging.getLogger('main').info("Start the job")
    print('cur_time is:', cur_time)
    # 读取实时输入数据
    logging.getLogger('main').info("Read real-time x data")
    x_real = data.get_x_real(cur_time)
    # 读取历史输入数据
    logging.getLogger('main').info("Read history x data")
    x_history = data.get_x_history(cur_time)
    # 读取历史样本数据
    logging.getLogger('main').info("Read history y data")
    y_history = data.get_y_history(cur_time)
    result = predict.predict(x_history, y_history, x_real)
    print(result)


if __name__ == '__main__':
    now = datetime.date.today()
    while now.weekday() > 5 or not is_workday(now):
        now -= datetime.timedelta(1)
    job(now)

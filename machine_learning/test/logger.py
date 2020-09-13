import logging
import sys

# 创建logger和handler，并设置日志文件路径
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(sys.path[0].split('stocks_predict.py')[0] + "/stocks_predict.log")
file_handler.setLevel(logging.DEBUG)
# 定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
# 给log添加handler
logger.addHandler(file_handler)


def info(name):
    def wrapper(func):
        def inner_wrapper(*args, **kwargs):
            try:
                re = func(*args, **kwargs)
                logging.getLogger("main").info(" {} executed successfully.".format(name))
                return re
            except Exception as e:
                logging.getLogger("main").error(" {} has ERROR in executing: {}".format(name, str(e)))

        return inner_wrapper

    return wrapper


if __name__ == '__main__':
    print(sys.path[0].split('main.py')[0] + "/main.log")

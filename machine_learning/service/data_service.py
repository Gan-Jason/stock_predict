import sys
from dao import stocks_dao
import properties


class DataService:
    def __init__(self):
        props = properties.Properties(sys.path[0].split('PFS_Back.py')[0] + "/data.properties")
        self.x_length = int(props.get("x_length"))
        self._length=int(props.get("y_length"))
        self.days=int(props.get("days"))


    def get_x_history(self, cur_time):
        target_time=





    def create_x_history_times(self,cur_time,days,length):
        history_times=[]
        

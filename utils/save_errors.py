import datetime
import time

from utils import read_parameters, save_parameters


def save_errors(msg):
    ts_current = time.mktime(datetime.datetime.now().timetuple())
    parameters = read_parameters.read_parameters()
    parameters['error'][str(ts_current).split('.')[0]] = msg
    save_parameters.save_parameters(parameters)

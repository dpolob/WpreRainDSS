import numpy as np
import datetime
import time
import pickle
import torch

from predictors.models import temp_model
from utils import data


def temp_prediction(ts_start):

    dt_start = datetime.datetime.fromtimestamp(ts_start)
    start = datetime.datetime(
                                year=dt_start.year,
                                month=dt_start.month,
                                day=dt_start.day,
                                hour=dt_start.hour,
                                minute=0,
                                second=0)
    # Get data from database
    status, result_data = data.get_data(start)
    if status == 'ERROR':
        return ('ERROR', result_data)
    status, result_data = data.get_hourly_data(result_data, 36, ts_start)
    if status == 'ERROR':
        return ('ERROR', result_data)
    # check integrity of returned values

    status, result_data = make_prediction(result_data)
    if status == 'ERROR':
        return ('ERROR', "Error in prediction")
    # convert to dictionary
    ts = time.mktime(start.timetuple())
    ts = int(ts) + (24*60*60)
    final_result = dict()
    for i in range(24):
        key = str(int(ts) - (3600*(i)))
        final_result[key] = result_data[i]

    return ("OK", final_result)


def make_prediction(lista):
    scaler = pickle.load(open('temp_scaler.pickle', 'rb'))
    x = np.array(lista)
    x = x.reshape(-1, 1)
    x = scaler.fit_transform(x)
    x = torch.from_numpy(x.reshape(1, -1))
    x = x.type(torch.float)

    net = temp_model.Red_Parcela(36, 24)
    net.load_state_dict(torch.load('temp.model.pt'))
    net.eval()
    y = net(x)

    y = y.data.numpy().reshape(-1, 1)
    y = scaler.inverse_transform(y)

    return ("OK", y.flatten().tolist())

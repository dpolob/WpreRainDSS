import numpy as np
import datetime
import time
import pickle

from predictors.models import temp_model
from predictors.rain.aemet import aemet

import torch


def rain_prediction():

    # Get data from AEMET database
    status, result_data = aemet.get_data_url()
    if status == 'ERROR':
        return ('ERROR', result_data)
    status, result_data = aemet.get_data(result_data)
    if status == 'ERROR':
        return ('ERROR', result_data)

    return ("OK", result_data)

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

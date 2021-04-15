import numpy as np
import datetime
import time
import pickle

from predictors.models import temp_model
from predictors.rain.aemet import aemet

#import torch


def rain_prediction():
    # Get data from AEMET database
    status, result_data = aemet.get_data_url()
    if status == 'ERROR':
        return ('ERROR', result_data, 0)
    status, result_data_probability, result_data_accumulated = aemet.get_data(result_data)
    if status == 'ERROR':
        return ('ERROR', result_data_probability, result_data_accumulated)
    
    return ("OK", result_data_probability, result_data_accumulated)

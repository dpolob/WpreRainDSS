"""
Functions for reading data from database and store it
"""
from utils import read_parameters
import pandas as pd
from flask import json
import requests
import datetime


def get_data(ts_start):
    # connect to cesens api_afc_enc_dss

    # load parameters
    parameters = read_parameters.read_parameters()

    response_cesens = requests.post(parameters['url_cesens'] + "/api/usuarios/login",
                                    data=json.dumps(parameters['login']),
                                    headers={"Content-Type": "application/json"})
    if not response_cesens.ok:
        return ("ERROR", "Cesens API for token is not working")

    token = "Token " + json.loads(response_cesens.text)["auth"]
    # retry last measures from ts_start -
    api_string = parameters['url_cesens'] + "/api/datos/" + parameters['station_id'] + "/" + str(parameters['metrics']['Temp']) + "/" + ts_start.strftime("%d%m%Y")
    response_cesens = requests.get(api_string, headers={"Content-Type": "application/json", "Authentication": token})
    if not response_cesens.ok:
        # update log
        return ("ERROR", "Cesens API for data is not working")

    return("OK", json.loads(response_cesens.text))


def get_hourly_data(data_dictionary, hours, ts_start):
    data = dict()
    for key, value in data_dictionary.items():
        if float(key) < float(ts_start):
            new_key = str(datetime.datetime.fromtimestamp(float(key)))
            data[new_key] = value

    temp_data = pd.DataFrame(list(data.items()), columns=['date', 'temp'])
    temp_data['dt_date'] = pd.to_datetime(temp_data['date'])
    temp_data.sort_values(by=['dt_date'], inplace=True, ascending=False)

    df_data = temp_data.set_index('dt_date').resample('1H').mean()

    # get only number of hour selected
    a = df_data['temp'].values.flatten()[0:hours].tolist()
    return ("OK", a)

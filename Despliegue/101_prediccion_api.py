import GLOBAL

import requests

from flask import Flask, jsonify, abort, request
from flask import make_response
from flask import json

import numpy as np
import pandas as pd
from datetime import datetime

from model import Red_General, Red_Parcela
import torch
import torch.nn as nn

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import pickle

app = Flask(__name__)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'not found'}), 404)

@app.route('/api/101_prediccion', methods=['GET'])
def get_general():
    if request.method == 'GET':
        data = request.get_json()
        fecha = data['fecha']
        data = '{"fecha": "' + str(data['fecha']) +'"}'
        response_cesens = requests.get(GLOBAL.URI_CESENS, data=data, headers={"Content-Type": "application/json"}, verify=False)
        if not response_cesens.ok:
            abort(404)
        response_general = requests.get(GLOBAL.URI_GENERAL, data=response_cesens.text, headers={"Content-Type": "application/json"}, verify=False)
        if not response_general.ok:
            abort(404)
        response_aemet = requests.get(GLOBAL.URI_AEMET, data=data, headers={"Content-Type": "application/json"}, verify=False)
        if not response_aemet.ok:
            abort(404)

        # red parcela
        temp_scaler = pickle.load(open(GLOBAL.MMSCALER, 'rb'))
        dato_general = temp_scaler.transform(np.fromstring(response_general.text.replace("[\n","").replace("\n","").replace("\n]",""), dtype=float, sep=",").reshape(1,-1))
        dato_aemet   = temp_scaler.transform(np.fromstring(response_aemet.text.replace("[\n","").replace("\n","").replace("\n]",""), dtype=float, sep=",").reshape(1,-1))

        lb = LabelBinarizer()
        lb.fit([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        fecha_mes = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S').month

        dato_mes = lb.transform(np.array(fecha_mes).reshape(1,-1)).astype(int)

        alt_scaler = pickle.load(open(GLOBAL.ALTSCALER, 'rb'))
        dato_altitud = alt_scaler.transform(np.array(GLOBAL.ALTITUDESTACION).reshape(1,-1))

        data = torch.cat((torch.tensor(dato_general, dtype=torch.float),
                          torch.tensor(dato_aemet, dtype=torch.float),
                          torch.tensor(dato_mes, dtype=torch.float),
                          torch.tensor(dato_altitud, dtype=torch.float)), dim=1)

        red_parcela = Red_Parcela((24 + 24 + 12 + 1), 24)
        red_parcela.load_state_dict(torch.load(GLOBAL.PESOMODELOPARCELA))
        red_parcela.eval()
        prediccion_parcela = red_parcela(data)
        prediccion_parcela = temp_scaler.inverse_transform(prediccion_parcela.data.numpy().reshape(1,-1))
        return jsonify(prediccion_parcela.tolist())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

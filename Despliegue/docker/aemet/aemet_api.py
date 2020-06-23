import GLOBAL
from flask import Flask, jsonify, abort, request
from flask import make_response
from flask import json
import numpy as np
import pandas as pd
from datetime import datetime

def conversion_horaria(infile):
    headers = ['id', 'fecha', 'temp']
    dtypes = {'id': 'int', 'fecha': 'str', 'temp': 'float'}
    parse_dates = ['fecha']
    df = pd.read_csv(infile, decimal=",", sep=";", header=None, names=headers, dtype=dtypes, parse_dates=parse_dates)
    df.index = df['fecha']
    df = df.resample('H').mean()
    return df


def aemet_bbdd(fecha, dataset):
    comienzo_dataset = datetime.strptime('2018-09-17 00:00:00', '%Y-%m-%d %H:%M:%S')
    fin_dataset = datetime.strptime('2019-09-16 23:59:59', '%Y-%m-%d %H:%M:%S')
    td_horas_fin = fin_dataset - datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    horas_fin = td_horas_fin.days * 24 + td_horas_fin.seconds // 3600
    td_horas= datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S') - comienzo_dataset
    horas = td_horas.days * 24 + td_horas.seconds // 3600
    if horas_fin < 24:
        horas = 24

    # leer dataset
    df = conversion_horaria(dataset)
    return df.iloc[horas + 1 : horas + 24 + 1 , 1].values

app = Flask(__name__)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/api/aemet', methods=['GET'])
def get_general():
    if request.method == 'GET':
        data = request.get_json()
        fecha = data['fecha']
        datos = aemet_bbdd(fecha, GLOBAL.AEMET)
        return jsonify(datos.tolist())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)

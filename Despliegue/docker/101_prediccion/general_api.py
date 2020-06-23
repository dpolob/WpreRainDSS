import GLOBAL
from flask import Flask, jsonify, abort, request
from flask import make_response
from flask import json
import numpy as np
import pandas as pd
from datetime import datetime
import pickle

from model import Red_General
import torch
import torch.nn as nn

app = Flask(__name__)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/api/general', methods=['GET'])
def general():
     if request.method == 'GET':
         data = request.get_json()
         temp_scaler = pickle.load(open(GLOBAL.MMSCALER, 'rb'))
         data = temp_scaler.transform(np.array(data['datos']).reshape(1,-1))
         data = torch.tensor(data, dtype=torch.float)

         red_general = Red_General(input_size=168, output_size=24, p_drop=0.5)
         red_general.load_state_dict(torch.load(GLOBAL.PESOMODELO))
         red_general.eval()

         prediccion_general = red_general(data)
         return jsonify(prediccion_general.detach().data.numpy().tolist())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)

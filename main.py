import time
import datetime

from flask import Flask, jsonify, request
from flask import Response, make_response

from utils import read_parameters, save_parameters, reset_parameters, save_errors
from predictors.rain import rain_prediction
from predictors.temp import temp_prediction

app = Flask(__name__)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'ALGORITHM not found'}), 404)


# API for rain predictions
@app.route('/api_afc_enc_wpre/t/<ts_start>', methods=['GET'])
def get_temp(ts_start):
    # if ts_start is a future date send error
    ts_start = int(ts_start)
    ts_current = time.mktime(datetime.datetime.now().timetuple())
    if ts_start > (ts_current + 60*60):  # future
        # update error's log
        save_errors.save_errors("Future time sent by user")
        return Response("{'ERROR' : 'Future time sent by user'}", status=400, mimetype='application/json')

     # if ts_start is in the past send NOT IMPLEMENTED YET
    if ts_start < (ts_current - 60 * 60):
        # update error's log
        save_errors.save_errors("Past predictions not implemented yet")
        return Response("{'ERROR' : 'Past predictions not implemented yet'}",
                        status=400, mimetype='application/json')

    else:
        # get temp predictions
        status, result = temp_prediction.temp_prediction(ts_start)
        if status == "ERROR":
            save_errors.save_errors(result)
            return Response(jsonify(dict({"ERROR": result})), status=400, mimetype='application/json')

        return make_response(jsonify(result), 200)

# API for rain predictions
@app.route('/api_afc_enc_wpre/r/<ts_start>', methods=['GET'])
def get_rain(ts_start): 
    ts_start = int(ts_start)
    ts_current = time.mktime(datetime.datetime.now().timetuple())
    # if ts_start is a future date send error
    if ts_start > (ts_current + 60 * 60):  # future
        # update error's log
        save_errors.save_errors("Future time sent by user")
        return Response("{'ERROR' : 'Future time sent by user'}",
                        status=400, mimetype='application/json')

    # if ts_start is in the past send NOT IMPLEMENTED YET
    if ts_start < (ts_current - 60 * 60):
        # update error's log
        save_errors.save_errors("Past predictions not implemented yet")
        return Response("{'ERROR' : 'Past predictions not implemented yet'}",
                        status=400, mimetype='application/json')

    # get rain predictions
    status, result = rain_prediction.rain_prediction(ts_start)
    if status == "ERROR":  # result is a string with error
        result_error = dict()
        result_error['msg'] = result
        save_errors.save_errors(result)
        return make_response(jsonify(result_error), 400)
    else:   # result is a dict with data
        return make_response(jsonify(result), 200)

# API for status
@app.route('/api_afc_enc_wpre/status', methods=['POST'])
def get_status():
    data = request.get_json()
    if data:  # data are sent, overwrite status file
        save_parameters.save_parameters(data)
        return Response("{'OK' : 'Status file updated'}", status=200,
                        mimetype='application/json')
    if not data:  # read parameters from file and serve
        result = {"OK" : read_parameters.read_parameters()}
        return make_response(jsonify(result), 200)

# API for reset
@app.route('/api_afc_enc_wpre/reset', methods=['GET'])
def get_reset():
    reset_parameters.reset_parameters()
    result = {"OK": "Reset Done"}
    return make_response(jsonify(result), 200)


if __name__ == '__main__':
    parameters=read_parameters.read_parameters()
    app.run(debug=True, host='0.0.0.0', port=parameters['flask_port'])

from os import execv
import time
import datetime
import utils

from flask import Flask, jsonify, request
from flask import Response, make_response

import globals
from utils import read_parameters, save_parameters, reset_parameters, save_errors
from predictors.rain import rain_prediction
from predictors.temp import temp_prediction

from multiprocessing.dummy import Pool
from send_mmt import send_mmt

# Logging module
import logging
import logging.config
import loggly.handlers

import exceptions

if globals.LOGLY:
    logging.config.fileConfig('loggly.conf')
else:
    logging.basicConfig(filename='mylog.log', level=logging.INFO)
logger = logging.getLogger()


pool = Pool(2)
app = Flask(__name__)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'status': 'RAIN PREDICTION route not found'}), 404)

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


# API for entry point commands
@app.route('/stop_alg', methods=['GET'])
def get_stopalg():
    if request.method == 'GET':
        logger.info("[{}] /stop_alg {} method requested from {}".format(globals.NAME, request.method, request.url))
        return Response("{\"status\" : \"STOPPED\", \"msg\" : {\"OK\" : \"Algorithm stopped\"}}", status=200, mimetype='application/json')

@app.route('/run_alg', methods=['POST'])
def post_runalg():
    global pool
    if request.method == 'POST':
        try:
            logger.info("/run_alg {} method requested from {}".format(request.method, request.url))
            data = request.get_json()
            if 'config' not in data.keys() or 'request_id' not in data.keys() or 'dss_api_endpoint' not in data.keys():
                raise exceptions.json_key_incorrect_exception(log=logger, value="[RAIN][/run_alg] Json Keys are wrong")
     
            status, result_probability, result_acummulated = rain_prediction.rain_prediction()

            logger.info("[RAIN][/run_alg] Status: {} Get %: {} and cummulated: {} from rain_predictor".format(status, result_probability, result_acummulated, logger))
            if status == "ERROR":
                raise exceptions.error_result_exception(log=logger, value="[RAIN][/run_alg] rain_prediction throwed an error")
            else:
                #convey to MMT through DSS
                print(result_probability, "%. Cummulated: " + str(result_acummulated) + " mm")
                pool.apply_async(send_mmt, (data, result_probability, "irrigation", "%. Cummulated: " + str(result_acummulated) + " mm"))
                # Reply to DSS 
                logger.info("/run_alg {} Reply to DSS. Algorithm started and info sent to MMT".format(request.method))
                return Response("{\"status\" : \"STARTED\", \"msg\" : {\"OK\" : \"Algorithm started and info sent to MMT\"}}", status=200,    mimetype='application/json')
                
        except exceptions.json_key_incorrect_exception as e:
            return make_response(jsonify(dict({"status" : "ERROR", "msg" : str(e)})), 500)
        except exceptions.error_result_exception as e:
            return make_response(jsonify(dict({"status" : "ERROR", "msg" : str(e)})), 500)
        except Exception as e:
            return make_response(jsonify(dict({"status" : "ERROR", "msg" : str(e)})), 500)

@app.route('/status_alg', methods=['GET'])
def get_statusalg():
    if request.method == 'GET':
        logger.info("[RAIN][/run_status] {} method requested from {}".format(globals.NAME, request.method, request.url))
        values = {"status": "STARTED",
                  "msg" : {
                            "flask_port" : globals.FLASKPORT,
                    }
                }
        return make_response(jsonify(values), 200)      

if __name__ == '__main__':
    #parameters=read_parameters.read_parameters()
    app.run(debug=True, host='0.0.0.0', port=globals.FLASKPORT)

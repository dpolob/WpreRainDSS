import requests
import urllib3
from flask import json

def send_mmt(data, result, observation, message):
    global logger
                #convey to MMT through DSS
    urllib3.disable_warnings()  # disable SSL warnings
    data_to_MMT= {
                            "request_id": data['request_id'],
                            "data_type": "number",
                            "data": [{
                                "timestamp": result['timestamp'],
                                "observation": observation,
                                "units": message,
                                "result": result['value']
                            }]
                        }
    try:
        a = requests.post(url=data['dss_api_endpoint'],
                                    data=json.dumps(data_to_MMT),
                                    headers= {"Content-Type": "application/json"},
                                    verify= False,  # disable SSL
                                    timeout = 4.0
                                    )
        a.raise_for_status()
        logger.info("/run_alg Convey to MMT success")
    except Exception as e:
        logger.info("/run_alg Convey to MMT failed {}".format(e))
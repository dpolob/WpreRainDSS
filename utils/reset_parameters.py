import pickle

parameters = {
    "status": "START",  # START or STOP select if algorithm runs
    "error": {},  # log for error, format { timestamp : "error message"}
    "flask_port": 5001,
    "last_check": 0,
    "url_cesens": "https://app.cesens.com",
    "login": {
        "nombre": "admin",
        "clave": "98b4t875"},
    "station_id": "2703",
    "metrics": {
        "Temp": 1,
        "RH": 6,
        "Solar": 21,
        "Soil": 30
        }
}


def reset_parameters():
    pickle.dump(parameters, open('/WeatherPrediction/status.pickle', 'wb'))


if __name__ == "__main__":
    reset_parameters()

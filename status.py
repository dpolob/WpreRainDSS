"""
File for global variables to command algorithm for weather predictions
"""

parameters = {
    "status": "START",  # START or STOP select if algorithm runs
    "error": {},  # log for error, format { timestamp : "error message"}
    "last_check": 0,
    "flask_port": 5001,
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

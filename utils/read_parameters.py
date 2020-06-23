"""
Function to load parameters
In a future status would be in a server
"""
import pickle


def read_parameters():
    parameters = pickle.load(open('/WeatherPrediction/status.pickle', 'rb'))
    return parameters

"""
Function to save parameters
In a future status would be in a server
"""

import pickle


def save_parameters(parameters):
    pickle.dump(parameters, open('/WeatherPrediction/status.pickle', 'wb'))

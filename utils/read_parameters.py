"""
Function to load parameters
"""
import pickle


def read_parameters():
    parameters = pickle.load(open('status.pickle', 'rb'))
    return parameters

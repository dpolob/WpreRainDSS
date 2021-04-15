## @package Errors
#  Handles exceptions from code

import logging

## json_key_incorrect_exception class
class json_key_incorrect_exception(Exception):
    def __init__(self,log, value):
        self.log = log
        self.value = value

    def __str__(self):
        if self.log:
            return(repr(self.value))
        else:
            return("Exception not defined")

## json_key_incorrect_exception class
class error_result_exception(Exception):
    def __init__(self,log, value):
        self.log = log
        self.value = value

    def __str__(self):
        if self.log:
            return(repr(self.value))
        else:
            return("Exception not defined")

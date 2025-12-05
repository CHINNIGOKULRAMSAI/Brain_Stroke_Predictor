import sys
from src.logger import logging

def error_message_details(error,error_details: sys):
    _,_,exc_info = error_details.exc_info()
    file_name = exc_info.tb_frame.f_code.co_filename
    error_message = "Error Occured in python scripts filename [{0}] and line number is [{1}], error: [{2}]".format(file_name,exc_info.tb_lineno,str(error))

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_details):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_details=error_details)


    def __str__(self):
        return self.error_message
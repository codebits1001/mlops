
import sys
from src.logger import logging

def get_error_message(error: Exception, error_detail: sys) -> str:
    
    _,_, exc_tb = error_detail.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
        return f"Error occured in script: [{file_name}] at line [{line_no}] with message: [str{error}]"
    else:
        return f"Error: {str(error)} (no traceback info available)"
    

# Creating the CustomException calss for mlops project
class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail:sys):
        super().__init__(error_message)

        self.error_message = get_error_message(error_message, error_detail)
    def __str__(self):
        return self.error_message
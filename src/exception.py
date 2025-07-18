import logging
import sys
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path,exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

def error_message_detail(error, error_detail: sys):
    """
    Custom error message to provide detailed information about the error.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in script: {file_name}, line: {exc_tb.tb_lineno}, error: {str(error)}"
    return error_message

class CustomException(Exception):
    """
    Custom exception class to handle exceptions with detailed error messages.
    """
    def __init__(self, error, error_detail: sys):
        super().__init__(error_message_detail(error, error_detail))
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message
    
if __name__ == "__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("Divide by zero error occurred") 
        raise CustomException(e, sys)
import logging
import os
import sys
from pathlib import Path
from typing import Union


def create_logger(logger_name: Union[Path, str] = None) -> logging.Logger:
    """
        Resource https://stackoverflow.com/questions/14058453/making-python-loggers-output-all-messages-to-stdout-in-addition-to-log-file
    """
    file_handler = logging.FileHandler(filename='tmp.log')
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )
    logger = logging.getLogger(logger_name or os.path.basename(__file__))
    return logger

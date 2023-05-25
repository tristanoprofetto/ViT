import os
import sys
import logging

from pythonjsonlogger import jsonlogger
from typing import Optional, List

json_formatter = jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s')
stream_handler = logging.StreamHandler(sys.stdout)

stream_handler.setFormatter(formatter)

logger = logging.getLogger()

logger.setLevel(logger.DEBUG)


class EnvVarsFilter(logging.Filter):
    """
    Filter for environment variables
    """

    def __init__(self, env_vars: List[str] = None):
        self.env_vars = env_vars

    
    def set_filter(self, record):
        for var in self.env_vars:
            record.__setattr__(os.environ.get(var))

        return True
    




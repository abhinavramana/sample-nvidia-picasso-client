import json
import os
from wombo_utilities import get_logger_for_file


logger = get_logger_for_file(__name__)


def log_environment_configs():
    environment_dict = {}
    for k, v in os.environ.items():
        environment_dict[k] = v
    pretty = json.dumps(environment_dict, indent=4)
    logger.info(f"Configs : {pretty}")

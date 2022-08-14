import os
import sys
import logging
import yaml
from easydict import EasyDict as edict
from datetime import datetime


def yaml_load(file_name):
    fc = None
    with open(file_name, 'r') as f:
        fc = edict(yaml.safe_load(f))

    return fc


def create_logger(log_dir: str, log_filename: str, logger_name: str):
    logger = logging.getLogger(logger_name)
    fhandler = logging.FileHandler(filename=os.path.join(log_dir, log_filename), mode='a')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)

    # redirecting console outputs to log file
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    return logger


def create_dir(dir_path):
    if os.path.isdir(dir_path):
        pass
    else:
        os.mkdir(dir_path)


def create_output_dir():
    job_dir = os.path.join('./outputs', f'{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    create_dir(job_dir)

    return job_dir


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


if __name__ == '__main__':
    pass

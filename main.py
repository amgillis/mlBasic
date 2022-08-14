# global imports
import logging
from datetime import datetime
from logging import getLogger
import pandas as pd

# local imports
from scripts.etl import run_etl
from scripts.prep import run_preprocessing
from scripts.model import run_model
from scripts.eval import run_eval
from utils import yaml_load, create_logger, create_output_dir


# creating output directory and log file
output_dir = create_output_dir()
log_filename = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logger = create_logger('./logs', log_filename, 'ml_logger')

# input config file path here
CONFIG_PATH = './config.yaml'
logger.info(f'loading config file from {CONFIG_PATH}')
config = yaml_load(CONFIG_PATH)

if __name__ == '__main__':
    df = run_etl(config, logger)
    Xtrain, ytrain, Xtest, ytest = run_preprocessing(df, config, logger)
    model, ypred = run_model(Xtrain, ytrain, Xtest, config, output_dir, logger)
    score, cr, cm = run_eval(ytest, ypred, output_dir, logger)

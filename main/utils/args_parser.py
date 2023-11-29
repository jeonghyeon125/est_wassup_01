from typing import Literal
import argparse

CONFIG_PATH_MAPPER = {
  'preprocess': './configs/preprocess_config.py',
  'train': './configs/config.py',
  'eval': './configs/config.py',
}

DESC_TITLE_MAPPER = {
  'preprocess': 'Preprocessing: Generate Dataset ',
  'train': 'Pytorch Train',
  'eval': 'Pytorch K-fold Cross Validation',
}

def get_args_parser(
  add_help=True,
  config_type: Literal['preprocess', 'train', 'eval']='train'
):
  parser = argparse.ArgumentParser(description=DESC_TITLE_MAPPER[config_type], add_help=add_help)
  parser.add_argument("-c", "--config", default=CONFIG_PATH_MAPPER[config_type], type=str, help="configuration file")
  return parser

# DEPRECATED
# def get_args_parser_dataset(add_help=True):
#   parser = argparse.ArgumentParser(description="Data preprocessing", add_help=add_help)
#   # inputs
#   parser.add_argument("--train-csv", default="./data/train.csv", type=str, help="train data csv file")
#   parser.add_argument("--test-csv", default="./data/test.csv", type=str, help="test data csv file")
#   # outputs
#   parser.add_argument("--output-train-feas-csv", default="./trn_X.csv", type=str, help="output train features")
#   parser.add_argument("--output-test-feas-csv", default="./tst_X.csv", type=str, help="output test features")
#   parser.add_argument("--output-train-target-csv", default="./trn_y.csv", type=str, help="output train targets")
#   # options
#   parser.add_argument("--index-col", default="Id", type=str, help="index column")
#   parser.add_argument("--target-col", default="SalePrice", type=str, help="target column")
#   parser.add_argument("--drop-cols", default=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], type=list, help="drop columns")
#   parser.add_argument("--fill-num-strategy", default="min", type=str, help="numeric column filling strategy (mean, min, max)")
#   parser.add_argument("--scaler", default="minmax", type=str, help="Choose one: ['minmax', 'standard', 'maxabs']")
#   parser.add_argument("--scaler", default="minmax", type=str, help="Choose one: ['minmax', 'standard', 'maxabs']")

#   return parser
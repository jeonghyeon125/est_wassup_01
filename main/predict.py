import copy
from utils import get_args_parser
from inferences import Predict

def main(train_cfg, prep_cfg):
  options = prep_cfg.get('options')
  output_csv = prep_cfg.get('output_data')
  Predict(
    index_col=options.get('index_col'),
    target_cols=options.get('target_cols'),
    target_drop_col=options.get('target_drop_col'),
    y_scaler_save=output_csv.get('y_scaler_save'),
    y_scaler=options.get('y_scaler'),
    **train_cfg
  ).run()

if __name__ == "__main__":
  train_args = get_args_parser().parse_args()
  config = {}
  exec(open(train_args.config).read())
  train_cfg = copy.deepcopy(config)
  
  preprocess_args = get_args_parser(config_type='preprocess').parse_args()
  exec(open(preprocess_args.config).read())
  
  main(train_cfg, config)
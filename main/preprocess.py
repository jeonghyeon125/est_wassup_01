from datasets import HomeData
import joblib
from utils import get_args_parser, create_path_if_not_exists

if __name__ == "__main__":
  args = get_args_parser(config_type='preprocess').parse_args()
  config = {}
  exec(open(args.config, encoding="utf-8").read())
  
  input_data = config.get('input_data')
  add_data = config.get('add_data')
  options = config.get('options')
  output_data = config.get('output_data')
  
  trn_X, trn_y, tst_X, y_scaler = HomeData(
    **input_data,
    **add_data,
    **options
  ).preprocess()
  
  for k,v in output_data.items():
    create_path_if_not_exists(v, True, '/')
  trn_X.to_csv(output_data.get('train_feas_csv'))
  tst_X.to_csv(output_data.get('test_feas_csv'))
  trn_y.to_csv(output_data.get('train_target_csv'))
  joblib.dump(y_scaler, output_data.get('y_scaler_save'))
import torch
from torch.utils.data import DataLoader
from utils import get_args_parser, create_path_if_not_exists, inference_test_ann
import numpy as np
import pandas as pd
from torch.utils.data.dataset import TensorDataset


def main(cfg):
  train_params = cfg.get('train_params')
  device = torch.device(train_params.get('device'))
  
  files = cfg.get('input_files')
  X_tst = pd.read_csv(files.get('tst_csv'), index_col=0)
  tensor_X_tst = torch.tensor(X_tst.to_numpy(dtype=np.float32))

  dl_params = train_params.get('data_loader_params')
  ds_tst = TensorDataset(tensor_X_tst)
  dl_tst = DataLoader(ds_tst, **dl_params)

  Model = cfg.get('model')
  model_params = cfg.get('model_params')
  model_params['input_dim'] = tensor_X_tst.shape[-1]
  model = Model(**model_params).to(device)
  print(model)

  output_files = cfg.get('output_files')
  model.load_state_dict(torch.load(output_files.get('output_train')))
  model.eval()
  
  result = inference_test_ann(dl_tst, model, device)
  
  test_id = X_tst.index.tolist()
  col_name = ['ID', 'ECLO']
  list_df = pd.DataFrame(zip(test_id, result), columns=col_name)
  list_df['ECLO'] = list_df['ECLO'].apply(lambda x: x if x >= 0 else 0)
  output_pred = output_files.get('output_pred')
  create_path_if_not_exists(output_pred)
  list_df.to_csv(output_pred, index=False)

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  exec(open(args.config).read())
  main(config)
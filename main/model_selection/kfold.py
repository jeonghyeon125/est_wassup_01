from dataclasses import dataclass, field
from makers import EvalMaker
from sklearn.model_selection import KFold
from torch import nn, manual_seed
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm.auto import trange
from trains import Evaluate, Train
from typing import Type

import pandas as pd


@dataclass
class KFoldCV:
  e_m: EvalMaker
  def run(self):
    manual_seed(2023)
    kfold = KFold(n_splits=self.e_m.n_split, shuffle=False)
    metrics = {'trn_rmse': [], 'val_rmse': []}
    X = self.e_m.get_X()
    y = self.e_m.get_y()
    for i, (trn_idx, val_idx) in enumerate(kfold.split(X)):
      self.e_m.init_model()
      X_trn, y_trn = X[trn_idx], y[trn_idx]
      X_val, y_val = X[val_idx], y[val_idx]

      ds_trn = TensorDataset(X_trn, y_trn)
      ds_val = TensorDataset(X_val, y_val)

      dl_trn = DataLoader(ds_trn, **self.e_m.dataloader_params)
      dl_val = DataLoader(ds_val, **self.e_m.dataloader_params)

      pbar = trange(self.e_m.epochs) #trange Tqdm + range
      for _ in pbar:
        Train.train_one_epoch(**{**self.e_m.get_train_parameters(), 'data_loader': dl_trn})
        trn_rmse = self.e_m.metric.compute().item()
        self.e_m.metric.reset()
        
        Evaluate.run(**{**self.e_m.get_eval_parameters(), 'data_loader': dl_val})
        val_rmse = self.e_m.metric.compute().item()
        self.e_m.metric.reset()
        pbar.set_postfix(trn_rmse=trn_rmse, val_rmse=val_rmse)
      metrics['trn_rmse'].append(trn_rmse)
      metrics['val_rmse'].append(val_rmse)
    return pd.DataFrame(metrics)

  def __call__(self):
    return self.run()
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from typing import Type
from .train_maker import TrainMaker


class PredMaker(TrainMaker):
  def __init__(
    self,
    index_col: str,
    target_cols: list=[],
    target_drop_col: str='',
    y_scaler_save: str='',
    y_scaler: Type[BaseEstimator]=None,
    **kwargs
  ):
    self.__y_scaler = y_scaler
    self.__y_scaler_save = y_scaler_save
    self.__index_col = index_col
    self.__target_cols = target_cols
    self.__target_drop_col = target_drop_col
    super().__init__(**kwargs)
  
  @property
  def y_origin_csv(self): return getattr(self, '__y_origin_csv')
  @property
  def y_output_csv(self): return getattr(self, '__output_pred')
  @property
  def y_scaler(self): return None if self.__y_scaler == None else joblib.load(self.__y_scaler_save)
  
  def get_idx_target_cols(self):
    """
    Returns:
        [index_col, target_cols[list]]
    """
    return [self.__index_col, self.__target_cols, self.__target_drop_col]
  
  def get_tst_X(self):
    """
    Returns:
        X_index, X_Dataloader
    """
    X_tst = pd.read_csv(getattr(self, '__tst_csv'), index_col=0)
    
    # X_tst.drop_duplicates(inplace=True)
    X_tst_tensor = torch.tensor(X_tst.to_numpy(dtype=np.float32))
    ds_tst = TensorDataset(X_tst_tensor)
    return X_tst.index.tolist(), DataLoader(ds_tst, **self.dataloader_params)
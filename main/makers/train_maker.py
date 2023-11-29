import numpy as np
import pandas as pd
import torch
from torch import nn, device
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from typing import Type


class TrainMaker():
  def __init__(self, model: Type[nn.Module], model_params: dict, **kwargs):
    self.__model_cls = model
    self.__model_params = model_params
    for _, params in kwargs.items():
      for k,v in params.items():
        setattr(self, f'__{k}', v)
    self.__set_dataloader()
    self.init_model()
    
  def __set_optimizer(self):
    self.__optimizer = getattr(self, '__optim')(self.model.parameters(), **getattr(self, '__optim_params'))

  def __set_dataloader(self):
    X_trn = torch.tensor(pd.read_csv(getattr(self, '__X_csv'), index_col=0).to_numpy(dtype=np.float32))
    y_trn = torch.tensor(pd.read_csv(getattr(self, '__y_csv'), index_col=0).to_numpy(dtype=np.float32))
    ds = TensorDataset(X_trn, y_trn)
    self.__dataloader = DataLoader(ds, **self.dataloader_params)
  
  def init_model(self):
    X_trn = torch.tensor(pd.read_csv(getattr(self, '__X_csv'), index_col=0).to_numpy(dtype=np.float32))
    self.__model_params['input_dim'] = X_trn.shape[-1]
    self.__model = self.__model_cls(**self.__model_params).to(self.device)
    self.__set_optimizer()
  
  @property
  def criterion(self): return getattr(self, '__loss')
  @property
  def device(self): return device(getattr(self, '__device'))
  @property
  def output_train(self): return getattr(self, '__output_train')
  @property
  def metric(self): return getattr(self, '__metric')
  @property
  def epochs(self): return getattr(self, '__epochs')
  @property
  def model(self): return self.__model
  @property
  def optimizer(self): return self.__optimizer
  @property
  def dataloader(self): return self.__dataloader
  @property
  def dataloader_params(self): return getattr(self, '__data_loader_params')
  
  def get_train_parameters(self):
    """
    Returns:
        { model, criterion, optimizer, dataloader, metric, device }
    """
    return {
      "model": self.model,
      "criterion": self.criterion,
      "optimizer": self.optimizer,
      "data_loader": self.dataloader,
      "metric": self.metric,
      "device": self.device
    }
import numpy as np
import pandas as pd
import torch
from torch import nn, device
from torchmetrics import MetricCollection
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from typing import Type
import joblib


class TrainMaker():
  def __init__(self, model: Type[nn.Module], model_params: dict, **kwargs):
    self.__model_cls = model
    self.__model_params = model_params
    for _, params in kwargs.items():
      for k,v in params.items():
        setattr(self, f'__{k}', v)
    self.__set_dataloader()
    self.init_model()
    self.init_metric_collection()
    
  def __set_optimizer(self):
    self.__optimizer = getattr(self, '__optim')(self.model.parameters(), **getattr(self, '__optim_params'))

  def __set_scheduler(self):
    if getattr(self, '__use_scheduler'):
      params = getattr(self, '__scheduler_params')
      self.__scheduler = getattr(self, '__scheduler_cls')(self.optimizer, **params)
    else:
      self.__scheduler = None

  def __set_dataloader(self):
    X_trn = torch.tensor(pd.read_csv(getattr(self, '__X_csv'), index_col=0).to_numpy(dtype=np.float32))
    y_trn = torch.tensor(pd.read_csv(getattr(self, '__y_csv'), index_col=0).to_numpy(dtype=np.float32))
    ds = TensorDataset(X_trn, y_trn)
    self.__dataloader = DataLoader(ds, **self.dataloader_params)
  
  def init_model(self):
    X_trn = torch.tensor(pd.read_csv(getattr(self, '__X_csv'), index_col=0).to_numpy(dtype=np.float32))
    self.__model_params['input_dim'] = X_trn.shape[-1]
    self.__loss_weight = torch.Tensor(getattr(self, '__loss_weight')).to(self.device)
    self.__model_params['output_dim'] = len(self.__loss_weight)
    self.__model = self.__model_cls(**self.__model_params).to(self.device)
    self.__set_optimizer()
    self.__set_scheduler()
  
  def init_metric_collection(self):
    metrics = MetricCollection(getattr(self, '__metrics'))
    self.__val_metrics = metrics.clone(prefix='val_').to(self.device)
    self.__trn_metrics = metrics.clone(prefix='trn_').to(self.device)
  
  def calc_multi_output_weight(self, output:torch.Tensor):
    return (output@self.loss_weight).item()
  
  @property
  def criterion(self): return getattr(self, '__loss')
  @property
  def device(self): return device(getattr(self, '__device'))
  @property
  def output_train(self): return getattr(self, '__output_train')
  @property
  def output_train_loss(self): return getattr(self, '__output_train_loss')
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
  @property
  def metrics(self): return {'val': self.__val_metrics, 'trn': self.__trn_metrics}
  @property
  def main_metric(self): return getattr(self, '__main_metric')
  @property
  def scheduler(self): return self.__scheduler
  @property
  def loss_weight(self): return self.__loss_weight
  @property
  def y_scaler(self): return None if getattr(self, '__y_scaler_path') == None else joblib.load(getattr(self, '__y_scaler_path'))
  
  def get_train_parameters(self):
    """
    Returns:
        { model, criterion, optimizer, dataloader, metrics[trn], device, scheduler, y_scaler }
    """
    return {
      "model": self.model,
      "criterion": self.criterion,
      "optimizer": self.optimizer,
      "data_loader": self.dataloader,
      "metrics": self.metrics['trn'],
      "device": self.device,
      "scheduler": self.scheduler,
      "y_scaler": self.y_scaler
    }
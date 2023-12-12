from dataclasses import dataclass
from makers import TrainMaker
import pandas as pd
import torch
from torch import nn, save
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from typing import Type
from tqdm.auto import trange
from sklearn.base import BaseEstimator
from utils.file_saver import create_path_if_not_exists


@dataclass
class Train:
  @classmethod
  def train_one_epoch(
    cls,
    model: Type[nn.Module],
    criterion: callable,
    optimizer: Type[Optimizer],
    data_loader: Type[DataLoader],
    device: str,
    metrics:Type[MetricCollection],
    scheduler:Type[LRScheduler],
    y_scaler: BaseEstimator = None
  ) -> None:
    '''train one epoch
    '''
    model.train()
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      output = model(X)
      loss = criterion(output, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if y_scaler != None:
        output = torch.tensor(y_scaler.inverse_transform(pd.DataFrame(output.cpu().detach()))).to(device)
        y = torch.tensor(y_scaler.inverse_transform(pd.DataFrame(y.cpu().detach()))).to(device)
      metrics.update(output, y)
    if scheduler != None: scheduler.step()
  
  @classmethod
  def run(cls, t_m: Type[TrainMaker]) -> None:
    trn_metrics = t_m.metrics['trn']
    trn_values = {k: '' for k in trn_metrics.keys()}
    trn_loss_save = {k: [] for k in trn_metrics.keys()}
    pbar = trange(t_m.epochs)
    for _ in pbar:
      train_params = t_m.get_train_parameters()
      cls.train_one_epoch(**train_params)
      for k,v in trn_metrics.compute().items():
        m = t_m.calc_multi_output_weight(v)
        trn_values[k] = m
        trn_loss_save[k].append(m)
      trn_metrics.reset()
      pbar.set_postfix(trn_values)
    create_path_if_not_exists(t_m.output_train)
    pd.DataFrame(trn_loss_save).to_csv(t_m.output_train_loss)
    save(t_m.model.state_dict(), t_m.output_train)
  
  @classmethod
  def __call__(cls, train_maker: Type[TrainMaker]) -> None:
    cls.run(train_maker)
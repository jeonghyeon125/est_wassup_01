from dataclasses import dataclass
from makers import TrainMaker
from torch import nn, save
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric
from typing import Type
from tqdm.auto import trange
from utils.file_saver import create_path_if_not_exists


@dataclass
class Train:
  @classmethod
  def train_one_epoch(cls, model: Type[nn.Module], criterion: callable, optimizer: Type[Optimizer], data_loader: Type[DataLoader], device: str, metric:Type[Metric]) -> None:
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
      metric.update(output, y)
  
  @classmethod
  def run(cls, t_m: Type[TrainMaker]) -> None:
    values = []
    pbar = trange(t_m.epochs)
    for _ in pbar:
      train_params = t_m.get_train_parameters()
      cls.train_one_epoch(**train_params)
      values.append(t_m.metric.compute().item())
      pbar.set_postfix(trn_loss=values[-1])
    create_path_if_not_exists(t_m.output_train)
    save(t_m.model.state_dict(), t_m.output_train)
  
  @classmethod
  def __call__(cls, train_maker: Type[TrainMaker]) -> None:
    cls.run(train_maker)
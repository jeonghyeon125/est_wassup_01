import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric
from typing import Type
from dataclasses import dataclass
from makers import EvalMaker

@dataclass
class Evaluate:
  @classmethod
  def run(cls, model: Type[nn.Module], data_loader: Type[DataLoader], metric: Type[Metric], device: str) -> None:
    '''evaluate
    
    Args:
        model: model
        data_loader: data loader
        device: device
        metrcis: metrics
    '''
    model.eval()
    with torch.inference_mode():
      for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        metric.update(output, y)
  
  @classmethod
  def __call__(cls, eval_maker: EvalMaker) -> None:
    cls.run(**eval_maker.get_eval_parameters())
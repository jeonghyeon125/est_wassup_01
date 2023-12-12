import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Type
from dataclasses import dataclass
from makers import EvalMaker
from sklearn.base import BaseEstimator

@dataclass
class Evaluate:
  @classmethod
  def run(cls, model: Type[nn.Module], data_loader: Type[DataLoader], metrics: dict, device: str, y_scaler: BaseEstimator) -> None:
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
        if y_scaler != None:
          output = torch.tensor(y_scaler.inverse_transform(pd.DataFrame(output.cpu()))).to(device)
          y = torch.tensor(y_scaler.inverse_transform(pd.DataFrame(y.cpu()))).to(device)
        metrics.update(output, y)
  
  @classmethod
  def __call__(cls, eval_maker: EvalMaker) -> None:
    cls.run(**eval_maker.get_eval_parameters())
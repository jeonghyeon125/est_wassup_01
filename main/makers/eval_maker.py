import numpy as np
import pandas as pd
import torch
from .train_maker import TrainMaker


class EvalMaker(TrainMaker):
  def __init__(
    self,
    **kwargs
  ):
    super().__init__(**kwargs)
  
  @property
  def n_split(self): return getattr(self, '__n_split')
  @property
  def output_eval(self): return getattr(self, '__output_eval')
  
  def get_X(self):
    return torch.tensor(pd.read_csv(getattr(self, '__X_csv'), index_col=0).to_numpy(dtype=np.float32))
  
  def get_y(self):
    return torch.tensor(pd.read_csv(getattr(self, '__y_csv'), index_col=0).to_numpy(dtype=np.float32))
  
  def get_eval_parameters(self):
    """
    return [ model, dataloader, metric, device ]
    """
    return {
      "model": self.model,
      "data_loader": self.dataloader,
      "metric": self.metric,
      "device": self.device
    }
    
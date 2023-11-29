from torch import sqrt
from torchmetrics import MeanSquaredLogError


class RootMeanSquaredLogError(MeanSquaredLogError):
  def __init__(self):
    super().__init__()
  
  def compute(self):
    print(sqrt(super().compute()))
    return sqrt(super().compute())
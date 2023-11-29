import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator
from tqdm.auto import tqdm
# from utils import CustomDataset
from collections import OrderedDict

########## 수정사항 ###############
# layer 수 + hidden dimension 수를 list 형태로 받는 ANN 설계
# activation func. 도 설계할 수 있으면 좋을 것 같음
# dropout 유무/ratio도 설정할 수 있도록 수정
##################################

activation_list = {"sigmoid": nn.Sigmoid(), "relu": nn.ReLU(), "tanh": nn.Tanh(), "prelu": nn.PReLU()}

class ANN(nn.Module):
  def __init__(self, input_dim: int=34, hidden_dim: list=[128, 128, 64, 32], activation: str="sigmoid", use_dropout: bool=False, drop_ratio: float=0.3):
    """ Artificial Neural Network(ANN) with Linear layers
      the model structure is like below:
      
      Linear
      Dropout
      Activation
      
      Linear
      Dropout
      ...
    
    
    Args:
      input_dim (int): dimension of input
      hidden_dim (list): list of hidden dimension. the length of 'hidden_dim' means the depth of ANN
      activation (str): activation name. choose one of [sigmoid, relu, tanh, prelu]
      use_dropout (bool): whether use dropout or not
      drop_ratio (float): ratio of dropout
    """
    super().__init__()
    dims = [input_dim] + hidden_dim 
    self.Identity = nn.Identity()
    self.dropout = nn.Dropout(drop_ratio)
    self.activation = activation_list[activation]

    model = [[nn.Linear(dims[i], dims[i+1]), self.dropout if use_dropout else self.Identity, self.activation] for i in range(len(dims) - 1)]
    output_layer = [nn.Linear(dims[-1], 1), nn.Identity()] # Relu는 항상 양수 값일 때 주는 방법
    self.module_list= nn.ModuleList(sum(model, []) + output_layer)
  def forward(self, x):
    for layer in self.module_list:
         x = layer(x)
    return x.squeeze(1)
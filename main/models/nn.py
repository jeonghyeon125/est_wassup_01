import torch
import torch.nn as nn
import torch.nn.functional as F

activation_list = {"sigmoid": nn.Sigmoid(), "relu": nn.ReLU(), "tanh": nn.Tanh(), "prelu": nn.PReLU()}

class ANN(nn.Module):
  def __init__(
    self,
    input_dim: int=5,
    hidden_dim: list=[128, 128, 64, 32],
    activation: str="sigmoid",
    use_drop:bool = True,
    drop_ratio: float=0.0,
    embed_cols_len: int = 0, # 차원 늘릴 원본 컬럼의 갯수
    embed_dim: int=10, # 늘릴 차원
    output_dim: int=1
  ):
    super().__init__()
    dims = [input_dim + (
        embed_cols_len * embed_dim -1 if embed_cols_len > 0 else 0
    )] + hidden_dim 
    self.Identity = nn.Identity()
    self.dropout = nn.Dropout(drop_ratio)
    self.activation = activation_list[activation]
    
    # Embed layer
    self.embed_layer = nn.Linear(1, embed_dim)
    self.embed_cols_len = embed_cols_len
    
    model = [[nn.Linear(dims[i], dims[i+1]), nn.BatchNorm1d(dims[i+1]), self.dropout if use_drop else self.Identity, self.activation] for i in range(len(dims) - 1)]
    output_layer = [nn.Linear(dims[-1], output_dim), nn.Tanh()] # 대구 교통사고 데이터 기준 음수 값이 나오는 경우가 있어서 제거
    # output_layer = [nn.Linear(dims[-1], output_dim), nn.Identity()]
    self.module_list= nn.ModuleList(sum(model, []) + output_layer)
  
  def forward(self, x):
    x_new = x[:, : x.shape[1] - self.embed_cols_len]
    for i in range(self.embed_cols_len):
      x_new = torch.concat([x_new, self.embed_layer(x[:,[-i]])], dim=1)
    # torch.set_printoptions(threshold=10_000)
    for layer in self.module_list:
        # print(layer)
        x_new = layer(x_new)
        # print(x_new)
    return x_new
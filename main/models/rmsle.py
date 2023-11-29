from torch import nn, sqrt, log


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return sqrt(self.mse(log(pred + 1), log(actual + 1)))

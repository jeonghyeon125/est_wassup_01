from torch import nn, Tensor

class MultiTaskMSELoss(nn.Module):
    def __init__(self, weights_per_task: list=[10,5,3,1]):
        super().__init__()
        self.task_num = len(weights_per_task)
        self.weights_per_task = weights_per_task
        self.mse = nn.MSELoss()

    def _custom_calc(self, pred:Tensor, actual:Tensor, task_num):
      if self.task_num == task_num: return 0
      mse = self.mse(pred[:,task_num], actual[:,task_num])
      mse = mse * self.weights_per_task[task_num]
      return mse + self._custom_calc(pred, actual, task_num+1)
    
    def forward(self, pred, actual):
      return self._custom_calc(pred, actual, 0)
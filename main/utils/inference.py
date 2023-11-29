from models import ANN
from torch.utils.data import DataLoader
import torch


def inference_test_ann(dl_tst: DataLoader, model: ANN, device: str="cpu"):
    result = []
    with torch.inference_mode():
      for X in dl_tst:
        X = X[0].to(device)
        output = model(X).squeeze().tolist()
        result.extend(output)
    return result
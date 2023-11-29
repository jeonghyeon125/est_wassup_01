import torch
from models import ANN, RMSLELoss
from metrics import RootMeanSquaredLogError


EXPERIMENT_NAME = 'experiment_2_epoch_50'

config = {
  'input_files': {
    'X_csv': './data/features/train_X.csv',
    'y_csv': './data/features/train_target.csv',
    'tst_csv': './data/features/test_X.csv',
  },
  'output_files': {
    'output_train': f'./histories/{EXPERIMENT_NAME}/model.pth',
    'output_eval': f'./histories/{EXPERIMENT_NAME}/eval.csv',
    'output_pred': f'./histories/{EXPERIMENT_NAME}/pred.csv'
  },
  'model': ANN,
  'model_params': {
    'input_dim': 'auto', # Always will be determined by the data shape
    'hidden_dim': [64, 64, 64],
    'use_drop': True,
    'drop_ratio': 0.7,
    'activation': 'relu',
  },
  'train_params': {
    'loss': RMSLELoss(),
    'optim': torch.optim.Adam,
    # 'metric': torchmetrics.MeanSquaredError(squared=False),
    'metric': RootMeanSquaredLogError(),
    'device': 'cpu',
    'epochs': 50,
    'data_loader_params': {
      'batch_size': 32,
    },
    'optim_params': {
      'lr': 0.001,
    },
  },
  'cv_params':{
    'n_split': 5,
  },
}
import torch
from torch import optim
from torchmetrics import (
  MeanSquaredError,
  MeanSquaredLogError
)
from torchmetrics.wrappers import MultioutputWrapper
from models import ANN
from loss import MultiTaskMSELoss, RMSLELoss

SCHEDULER = {
  'CAWarmRestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts, # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
  'CALR': optim.lr_scheduler.CosineAnnealingLR, # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
  'lambdaLR': optim.lr_scheduler.LambdaLR, # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
}

# WARNING: EMBEDDING_COLS 가 preprocess_config 의 COLS 와 같은지 확인하세요.
EMBEDDING_COLS = []
EMBEDDING_NODE = 10 # 늘릴 차원 수

# CUSTOM_WEIGHT = [10, 5, 3, 1]
CUSTOM_WEIGHT = [1]
NUM_OF_TASKS = len(CUSTOM_WEIGHT)

CUSTOM_LOSS = {
  # multi_task loss 사용법
  # weights_per_task = 각 target col 에 대한 (loss 에 곱셈할) 가중치를 입력해주세요. e.g. [10,5,3,1]
  # 가중치는 preprocess_config 에 Target_cols 순서대로 입력해주세요.
  # 만약 target_col 의 길이가 1 이라면 그냥 [1] 이렇게 주면 됩니다.
  'multi_task_mse': MultiTaskMSELoss,
  'rmsle': RMSLELoss
}

EXPERIMENT_NAME = 'experiment_9'

config = {
  'input_files': {
    'X_csv': './data/features/train_X.csv',
    'y_csv': './data/features/train_target.csv',
    'tst_csv': './data/features/test_X.csv'
  },
  'output_files': {
    'output_train': f'./histories/{EXPERIMENT_NAME}/model.pth',
    'output_eval': f'./histories/{EXPERIMENT_NAME}/eval.csv',
    'output_pred': f'./histories/{EXPERIMENT_NAME}/pred.csv'
  },
  'model': ANN,
  'model_params': {
    'input_dim': 'auto', # Always will be determined by the data shape
    'hidden_dim': [128, 64],
    'use_drop': True,
    'drop_ratio': 0.7,
    'activation': 'relu',
    
    # Embedding params
    # embed_cols_len 은 만약 embedding 을 원치 않을 경우, 0으로 설정해주세요.
    'embed_cols_len': len(EMBEDDING_COLS), # preprocess_config 에서 설정한 column 갯수
    'embed_dim': EMBEDDING_NODE # 늘릴 차원의 수
  },
  'train_params': {
    # 'loss': RMSLELoss(),
    'use_scheduler': True,
    'scheduler_cls': SCHEDULER['CALR'],
    'scheduler_params': {
      # 'T_0': 200,
      # 'T_mult': 2,
      'T_max': 10,
      'eta_min': 0.00001
    },
    # 'loss': torch.nn.MSELoss(),
    'loss': CUSTOM_LOSS['multi_task_mse'](weights_per_task=CUSTOM_WEIGHT),
    'loss_weight': CUSTOM_WEIGHT,
    'optim': torch.optim.Adam,
    'main_metric': 'rmse',
    'metrics': {
      'rmse': MultioutputWrapper(MeanSquaredError(squared=False), NUM_OF_TASKS),
      'mse': MultioutputWrapper(MeanSquaredError(), NUM_OF_TASKS),
      # 'msle': MultioutputWrapper(MeanSquaredLogError(), NUM_OF_TASKS)
    },
    'device': 'cuda:0',
    'epochs': 20,
    'data_loader_params': {
      'batch_size': 32,
    },
    'optim_params': {
      'lr': 0.01,
    },
  },
  'cv_params':{
    'n_split': 5,
  }
}
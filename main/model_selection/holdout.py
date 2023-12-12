from dataclasses import dataclass
from makers import EvalMaker
from torch import manual_seed
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm.auto import trange
from trains import Evaluate, Train
import pandas as pd


@dataclass
class HoldoutWithTestCSV:
  e_m: EvalMaker
  def run(self):
    manual_seed(2023)
    main_metric = self.e_m.main_metric
    trn_main_metric = f'trn_{main_metric}'
    val_main_metric = f'val_{main_metric}'
    metrics = {trn_main_metric: [], val_main_metric: [], 'trn_history': [], 'val_history': []}
    
    trn_metrics = self.e_m.metrics['trn']
    val_metrics = self.e_m.metrics['val']
    
    X = self.e_m.get_X()
    y = self.e_m.get_y()
    
    self.e_m.init_model()

    ds_trn = TensorDataset(X, y)
    ds_val = TensorDataset(X, y)

    dl_trn = DataLoader(ds_trn, shuffle=True, **self.e_m.dataloader_params)
    dl_val = DataLoader(ds_val, **self.e_m.dataloader_params)
      
    pbar = trange(self.e_m.epochs) #trange Tqdm + range
    trn_values = {k: [] for k in trn_metrics.keys()}
    val_values = {k: [] for k in val_metrics.keys()}
    for _ in pbar:
      Train.train_one_epoch(**{**self.e_m.get_train_parameters(), 'data_loader': dl_trn})
      for k,v in trn_metrics.compute().items(): trn_values[k].append(self.e_m.calc_multi_output_weight(v))
      trn_metrics.reset()
      
      Evaluate.run(**{**self.e_m.get_eval_parameters(), 'data_loader': dl_val})
      for k,v in val_metrics.compute().items(): val_values[k].append(self.e_m.calc_multi_output_weight(v))
      val_metrics.reset()
      
      pbar.set_postfix({trn_main_metric: trn_values[trn_main_metric][-1], val_main_metric: val_values[val_main_metric][-1]})
    metrics['trn_history'].append(trn_values)
    metrics['val_history'].append(val_values)
    metrics[trn_main_metric].append(trn_values[trn_main_metric][-1])
    metrics[val_main_metric].append(val_values[val_main_metric][-1])
    return pd.DataFrame(metrics)

  def __call__(self):
    return self.run()
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from typing import Any, Type
from makers import PredMaker
from utils import create_path_if_not_exists


class Predict():
  def __init__(
    self, 
    index_col: str,
    target_cols: list=[],
    target_drop_col: str='',
    y_scaler_save: str='',
    y_scaler: Type[BaseEstimator]=None,
    **kwargs: Any
  ):
    self.p_m = PredMaker(index_col, target_cols, target_drop_col, y_scaler_save, y_scaler, **kwargs)
  
  def inference_test_ann(self, dl_tst):
    result = []
    with torch.inference_mode():
      for X in dl_tst:
        # TODO: multi task features should be added
        X = X[0].to(self.p_m.device)
        output = self.p_m.model(X).squeeze().tolist()
        result.extend(output)
    return result

  def unnormalization(self, pred:pd.DataFrame):
    scaler = self.p_m.y_scaler
    # TODO: additional feature when y_scaler is a python function
    if scaler == None:
      return pred
    return scaler.inverse_transform(pred)
  
  def run(self):
    self.p_m.model.load_state_dict(torch.load(self.p_m.output_train))
    self.p_m.model.eval()
    
    # for param in self.p_m.model.parameters():
    #   print(param.data)
    
    X_tst_index, X_tst_dl = self.p_m.get_tst_X()
    result = self.inference_test_ann(X_tst_dl)
    id, target_cols, target_drop_col = self.p_m.get_idx_target_cols()
    list_df = pd.DataFrame(result, columns=target_cols)
    list_df = pd.concat([pd.DataFrame({id: X_tst_index}), list_df], axis=1)
    
    list_df[target_cols] = self.unnormalization(list_df[target_cols])
    list_df[target_cols] = list_df[target_cols].fillna(0)
    # list_df[target_cols] = list_df[target_cols].map(lambda x: int(np.round(x)))
    
    if len(target_cols) > 1 and target_drop_col != None:
      loss_weight = self.p_m.loss_weight
      final_result = list_df[target_cols[0]] * loss_weight[0].item()
      for i in range(1, len(target_cols)):
        final_result += list_df[target_cols[i]] * loss_weight[0].item()
      
      list_df.drop(columns=target_cols, inplace=True)
      list_df[target_drop_col] = final_result
    
    create_path_if_not_exists(self.p_m.y_output_csv)
    list_df.reset_index(drop=True).to_csv(self.p_m.y_output_csv, index=False, encoding='cp949')
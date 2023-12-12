from .dataset import Dataset

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Literal, List
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from .utils import custom_X_preprocess_cat


@dataclass(kw_only=True)
class HomeData(Dataset):
    fill_num_strategy: Literal["mean", "min", "max"] = "min"
    x_scaler: BaseEstimator = None
    y_scaler: BaseEstimator = None
    embedding_cols: list = field(default_factory=list)

    def _embedding_X(self, X_df: pd.DataFrame):
        df_embed = X_df[self.embedding_cols].copy()
        X_df.drop(columns=self.embedding_cols, inplace=True)
        for col in self.embedding_cols:
            le = LabelEncoder()
            df_embed[col] = le.fit_transform(df_embed[col])
        return df_embed

    def _scale_X(self, X_df: pd.DataFrame):
        self.x_scaler.fit(X_df)
        return self.x_scaler.transform(X_df).astype(dtype=np.float32)

    def _scale_Y(self, target: iter):
        # TODO: additional feature when y_scaler is a python function
        if self.y_scaler == None:
            return pd.DataFrame(target)
        self.y_scaler.fit(target)
        return pd.DataFrame(
            self.y_scaler.transform(target).astype(dtype=np.float32),
            columns=self.y_scaler.get_feature_names_out(),
        )

    def _X_preprocess(self, X_df: pd.DataFrame, add_df_list: list[pd.DataFrame]):
        idx = X_df.index
        # Custom X preprocess for cat data - label encoded or cat objects
        X_df = custom_X_preprocess_cat(X_df, add_df_list)
        
        if '동' not in self.embedding_cols:
            X_df.drop(columns=['동'], inplace=True)
        
        # Embedding - Label Encoding
        df_embed = self._embedding_X(X_df).reset_index(drop=True)

        # Numeric
        df_num = X_df.select_dtypes(include=["number"])
        if self.fill_num_strategy == "mean":
            fill_values = df_num.mean(axis=1)
        elif self.fill_num_strategy == "min":
            fill_values = df_num.min(axis=1)
        elif self.fill_num_strategy == "max":
            fill_values = df_num.max(axis=1)
        df_num.fillna(fill_values, inplace=True)
        df_num.reset_index(drop=True, inplace=True)
        if self.x_scaler is not None:
            df_num = pd.DataFrame(self._scale_X(df_num), columns=df_num.columns)

        # Categorical
        df_cat = X_df.select_dtypes(include=["object"])
        enc = OneHotEncoder(
            dtype=np.float32,
            sparse_output=False,
            drop="if_binary",
            handle_unknown="ignore",
        )
        enc.fit(df_cat)
        df_cat_onehot = pd.DataFrame(
            enc.transform(df_cat), columns=enc.get_feature_names_out()
        ).reset_index(drop=True)

        return pd.concat([df_num, df_cat_onehot, df_embed], axis=1).set_index(
            idx
        )

    def preprocess(self):
        """
        Returns:
            trn_X, target, tst_X, add_df_list

        Args:
            add_df_list: List of dataframes of additional data
        """
        trn_df, target, tst_df, add_df_list = self._get_dataset()

        # X Features
        trn_X = self._X_preprocess(trn_df, add_df_list)
        tst_X = self._X_preprocess(tst_df, add_df_list)

        # Y Feature
        target = self._scale_Y(target)

        print(trn_X.shape)
        print(tst_X.shape)
        return trn_X, target, tst_X, self.y_scaler

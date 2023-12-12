from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(kw_only=True)
class Dataset:
    __DEFAULT_USE_COLS = ("사고일시", "요일", "기상상태", "시군구", "도로형태", "노면상태", "사고유형")

    __DEFAULT_DROP_COLS = ()

    train_csv: str
    test_csv: str
    add_csv_list: list[str]
    index_col: str
    target_cols: list
    target_drop_col: str
    drop_cols: tuple[str] = __DEFAULT_DROP_COLS
    use_cols: tuple[str] = __DEFAULT_USE_COLS
    ignore_drop_cols: bool = True

    def _read_df(self, split: Literal["train", "test", "add"] = "train"):
        if split == "train":
            df = pd.read_csv(self.train_csv, index_col=self.index_col)
            df.dropna(axis=0, subset=self.target_cols, inplace=True)
            target = df[self.target_cols]
            df.drop(self.target_cols, axis=1, inplace=True)
            if len(self.target_cols) > 1:
                df.drop([self.target_drop_col], axis=1, inplace=True)
            return df, target
        elif split == "test":
            df = pd.read_csv(self.test_csv, index_col=self.index_col)
            return df
        elif split == "add":
            df_list = [pd.read_csv(add_csv, index_col=0) for add_csv in self.add_csv_list]
            return df_list
        raise ValueError(f'"{split}" is not acceptable.')

    def _get_dataset(self):
        """
        Get filtered dataset.
        Drop columns `drop_cols` when `ignore_drop_cols` set False
        Use columns `use_cols` when `ignore_drop_cols` set True

        Returns:
            (trn_df, target, tst_df)
        """
        trn_df, target = self._read_df("train")
        tst_df = self._read_df("test")
        add_df_list = self._read_df("add")

        if self.ignore_drop_cols:
            # use `use_cols`
            trn_df = trn_df[self.use_cols]
            tst_df = tst_df[self.use_cols]
        else:
            # drop `drop_cols`
            trn_df.drop(self.drop_cols, axis=1)
            tst_df.drop(self.drop_cols, axis=1)

        return (trn_df, target, tst_df, add_df_list)

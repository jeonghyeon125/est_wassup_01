from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler


DROP_X_COLS = []

USE_X_COLS = ['사고일시', '요일', '기상상태', '시군구', '도로형태', '노면상태', '사고유형']
# USE_X_COLS = ['사고일시', '요일', '기상상태', '도로형태', '노면상태', '사고유형']

SCALER = {
  "standard": StandardScaler,
  "minmax": MinMaxScaler,
  "maxabs": MaxAbsScaler
}

config = {
  'input_data': {
    'train_csv': './data/origin/train.csv',
    'test_csv': './data/origin/test.csv'
  },
  'output_data': {
    'train_feas_csv': './data/features/train_X.csv',
    'test_feas_csv': './data/features/test_X.csv',
    'train_target_csv': './data/features/train_target.csv'
  },
  'options': {
    'index_col': 'ID',
    'target_col': 'ECLO',
    'ignore_drop_cols': True, # 'use_cols' is ignored when False, 'drop_cols' are ignored when True
    'drop_cols': DROP_X_COLS,
    'use_cols': USE_X_COLS,
    'fill_num_strategy': 'min', # choose one: ['min', 'mean', 'max']
    # 'x_scaler': SCALER['minmax'](), # put None for no X scaling process
    'x_scaler': None
  }
}
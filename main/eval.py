from makers import EvalMaker
from model_selection import KFoldCV, HoldoutWithTestCSV
import pandas as pd
from utils import get_args_parser, create_path_if_not_exists


if __name__ == "__main__":

  args = get_args_parser().parse_args()
  
  config = {}
  exec(open(args.config).read())

  eval_maker = EvalMaker(**config)
  cv = KFoldCV(eval_maker)
  # cv = HoldoutWithTestCSV(eval_maker)
  res = cv()
  res = pd.concat([res, res.iloc[:,:2].apply(['mean'])])
  # res = pd.concat([res, res.iloc[:,:2].apply(['mean', 'std'])])
  print(res)
  create_path_if_not_exists(eval_maker.output_eval)
  res.to_csv(eval_maker.output_eval)
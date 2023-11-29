from torch import manual_seed
from trains import Train
from makers import TrainMaker
from utils import get_args_parser


if __name__ == "__main__":
  args = get_args_parser().parse_args()
  config = {}
  exec(open(args.config).read())
  manual_seed(2023)
  train_maker = TrainMaker(**config)
  Train.run(train_maker)
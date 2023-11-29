import os


def create_path_if_not_exists(path: str, remove_filename: bool=True, split_by: str='/') -> None:
    if remove_filename:
       path = get_dirs_only(path, split_by)
    os.makedirs(path, exist_ok=True)

def get_dirs_only(path: str, split_by: str='/') -> str:
  arr = path.split(split_by)[:-1]
  return split_by.join(arr)
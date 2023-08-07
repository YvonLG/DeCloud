


if __name__ == '__main__':

    from datasets.datasets import TimeSerieDS
    from datasets.utils import get_path_dicts
    from pathlib import Path
    import numpy as np

    path_dicts = get_path_dicts(Path('data/BIHAR_2022_2023_test'), sanity_check=True)

    path_dicts_train = path_dicts
    path_dicts_valid = path_dicts

    TimeSerieDS(path_dicts_train, ts_len=3, sel_mode='one').save('train_dataset.pickle')
    TimeSerieDS(path_dicts_valid, ts_len=3, sel_mode='one').save('valid_dataset.pickle')
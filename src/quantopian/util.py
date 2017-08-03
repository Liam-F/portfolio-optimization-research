import pandas as pd
import random


def sample_file(filepath, sample_size):
    nb_rows = sum(1 for line in open(filepath)) - 1
    skip = sorted(random.sample(range(1, nb_rows + 1), nb_rows - sample_size))
    return pd.read_csv(filepath, parse_dates=True, index_col=0, skiprows=skip)
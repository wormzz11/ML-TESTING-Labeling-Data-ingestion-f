import os 
import pandas as pd
def load_data(PATH):
    return pd.read_csv(PATH)


def merge_training_data():
    pass

def append_csv(subset, path):
    subset.to_csv(path, mode='a', index=False, header=not os.path.exists(path))


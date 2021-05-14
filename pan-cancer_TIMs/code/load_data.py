import pandas as pd
import numpy as np
import os
import anndata

data_dir = "../data/"


def read_dataset(name):
    data = pd.read_csv(name+"_normalized_expression.csv")
    counts = data.drop(['index'], axis=1)
    metadata = pd.read_csv(name+"_metadata.csv")
    adata = anndata.AnnData(X=counts, obs=metadata)
    return adata


def get_valid_datasets(dir_path):
    files = os.listdir(dir_path)
    lst = []
    rtn = []
    for f in files:
        if suffix(f, suf='_metadata.csv'):
            lst.append(suffix(f, suf='_metadata.csv'))
        if suffix(f, suf="_normalized_expression.csv"):
            lst.append(suffix(f, suf="_normalized_expression.csv"))
    while lst:
        curr = lst.pop()
        if os.path.exists(dir_path+curr+"_metadata.csv") and os.path.exists(dir_path+curr+"_metadata.csv"):
            rtn.append(curr)
            lst.remove(curr)
    return rtn


def suffix(word, suf):
    if word[-len(suf):] == suf:
        return word[:-len(suf)]
    return False


def load_data(data_dir):
    rtn = []
    datasets = get_valid_datasets(data_dir)
    print(f" found {len(datasets)} datasets in {data_dir}")
    for name in get_valid_datasets(data_dir):
        rtn.append(read_dataset(os.path.join(data_dir, name)))
        print(f" added {name} dataset")
    return rtn

if __name__ == "__main__":
    data = load_data(data_dir)
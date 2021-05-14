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
    return rtn


def suffix(word, suf):
    if word[-len(suf):] == suf:
        return word[:-len(suf)]
    return False


def load_data():
    rtn = []
    for name in get_valid_datasets(data_dir):
        rtn.append(read_dataset(name))
    return rtn
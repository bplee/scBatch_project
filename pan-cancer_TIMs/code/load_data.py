import pandas as pd
import numpy as np
import os
import anndata
import sys
import scanpy as sc

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")
from Step0_Data.code.starter import *

data_dir = "../data/"
TIM_DATA_FILEPATH = "/data/leslie/bplee/scBatch/pan-cancer_TIMs/quickload_data/TIMs_all_data.h5ad"


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


def quick_load(filepath=TIM_DATA_FILEPATH):
    return anndata.read_h5ad(filepath)

# def cross_tumor_filter

# def filter_cancers(cancers_types_to_remove=[])

if __name__ == "__main__":
    adata = quick_load(TIM_DATA_FILEPATH)
    print(f" loaded all TIM data into anndata obj named: `data`")
import anndata
import sys
import os
import scanpy as sc
import pandas as pd

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

# this data is from: https://zenodo.org/record/4263972#.YFjtJS1h1B0


TIC_DATA_PATH = "/data/leslie/bplee/scBatch/TIC_atlas/data/TICAtlas.h5ad"
TIC_META_DATA_PATH = "/data/leslie/bplee/scBatch/TIC_atlas/data/TICAtlas_metadata.csv"
TIC_DOWNSAMPLED_PATH = "/data/leslie/bplee/scBatch/TIC_atlas/data/TICAtlas_downsampled_1000.h5ad"

def load_data(data_path=TIC_DATA_PATH):
    return anndata.read_h5ad(data_path)

def load_meta_data(meta_data_path=TIC_META_DATA_PATH):
    return pd.read_csv(meta_data_path, index_col=0)

def load_downsampled(data_path=TIC_DOWNSAMPLED_PATH):
    return anndata.read_h5ad(data_path)

def remove_low_count_patients(adata):


def clean_tic(adata):

    return adata

if __name__ == "__main__":
    adata = load_data()
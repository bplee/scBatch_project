import anndata
import sys
import os
import scanpy as sc
import numpy as np
import pandas as pd

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

from Step0_Data.code.starter import *

COUNTS_FILEPATH = "/data/leslie/bplee/scBatch/broad_rcc/data/SCP1288/expression/ccRCC_scRNASeq_NormalizedCounts.txt"
METADATA_FILEPATH = "/data/leslie/bplee/scBatch/broad_rcc/data/SCP1288/metadata/Final_SCP_Metadata.txt"
H5AD_FILEPATH = "/data/leslie/bplee/scBatch/broad_rcc/data/SCP1288/quickload_data/ccRCC_broad_normalized_counts.h5ad"


def load_data(counts_path=COUNTS_FILEPATH, metadata_path=METADATA_FILEPATH):
    # this is read in transposed (cells as columns)
    adata = anndata.read_text(counts_path).T
    meta = pd.read_csv(metadata_path, sep="\t", index_col=0)
    # the first row is just a type
    adata.obs = meta.iloc[1:, :]
    return adata


def quick_load(h5ad_fp = H5AD_FILEPATH):
    adata = anndata.read_h5ad(h5ad_fp)
    return adata


# def filter_broad_data(adata):



if __name__ == "__main__":
    adata = quick_load()
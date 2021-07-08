import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import os


data_dir="/lila/data/leslie/bplee/scBatch_project/sclc_peer/data/"
quickload="/lila/data/leslie/bplee/scBatch_project/sclc_peer/quickload_data/adata_sclc_only.h5ad"

def quickload_sclc(filepath=quickload):
    return anndata.read_h5ad(filepath)


if __name__ == "__main__":
    adata = quickload_sclc()
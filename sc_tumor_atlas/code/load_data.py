import anndata
import sys
import os

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")


TIC_DATA_PATH = "/data/leslie/bplee/scBatch/sc_tumor_atlas/data/TICatlas.h5ad"

def load_data(data_path=TIC_DATA_PATH):
    return anndata.read_h5ad(data_path)

def clean_tic(adata):
    

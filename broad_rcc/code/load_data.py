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
METADATA_FILEPATH = "/data/leslie/bplee/scBatch/broad_rcc/data/SCP1288/metadata/ccRCC_scRNASeq_NormalizedCounts.txt"

def load_data(counts_path=COUNTS_FILEPATH, metadata_path=METADATA_FILEPATH):
    adata = anndata.read_text(counts_path)
    adata.obs = pd.read_csv(metadata_path, sep="\t")
    return adata

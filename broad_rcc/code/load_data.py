import anndata
import sys
import os
import scanpy as sc
import numpy as np
import pandas as pd

WORKING_DIR = "/data/leslie/bplee/scBatch_project"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

from Step0_Data.code.starter import *

COUNTS_FILEPATH = "/data/leslie/bplee/scBatch_project/broad_rcc/data/SCP1288/expression/ccRCC_scRNASeq_NormalizedCounts.txt"
METADATA_FILEPATH = "/data/leslie/bplee/scBatch_project/broad_rcc/data/SCP1288/metadata/Final_SCP_Metadata.txt"
H5AD_FILEPATH = "/data/leslie/bplee/scBatch_project/broad_rcc/quickload_data/ccRCC_broad_normalized_counts.h5ad"

domain_name = "donor_id"
label_name = "FinalCellType"

# PATIENTS: `donor_id`
# ____________
# P90     8426
# P76     7912
# P915    6541
# P55     4637
# P913    3744
# P906    2470
# P916     316
# P912     280

# CELL TYPES: `FinalCellType`
# _____________________________
# 41BB-Lo CD8+ T cell      5420
# TP2                      4599
# TP1                      3324
# FOLR2-Hi TAM             1528
# MitoHigh CD8+ T cell     1482
# MitoHigh Myeloid         1407
# Effector T-Helper        1389
# GPNMB-Hi TAM             1382
# 41BB-Hi CD8+ T cell      1321
# MitoHigh T-Helper        1316
# FGFBP2- NK               1306
# VSIR-Hi TAM              1070
# B cell                    962
# CD16- Monocyte            844
# NKT                       811
# T-Reg                     750
# Cycling CD8+ T cell       701
# LowLibSize Macrophage     672
# Memory T-Helper           579
# FGFBP2+ NK                493
# Plasma cell               463
# MitoHigh NK               446
# CD16+ Monocyte            313
# CD1C+ DC                  308
# Misc/Undetermined         278
# Endothelial               271
# CXCL10-Hi TAM             226
# Cycling TAM               175
# MX1-Hi CD8+ T cell        132
# Cycling Tumor             117
# CLEC9A+ DC                111
# Fibroblast                 91
# Mast cell                  39


def broad_load_data(counts_path=COUNTS_FILEPATH, metadata_path=METADATA_FILEPATH):
    # this is read in transposed (cells as columns)
    adata = anndata.read_text(counts_path).T
    meta = pd.read_csv(metadata_path, sep="\t", index_col=0)
    # the first row is just a type
    adata.obs = meta.iloc[1:, :]
    return adata


def broad_quick_load(h5ad_fp = H5AD_FILEPATH):
    adata = anndata.read_h5ad(h5ad_fp)
    return adata


def filter_patients(adata, patients_to_remove=["P916", "P912"]):
    bool_inds = ~adata.obs.donor_id.isin(patients_to_remove)
    print(f'removing {sum(~bool_inds)} cells')
    return adata[bool_inds, :]

low_cell_types = ["Misc/Undetermined",
                  "Endothelial",
                  "CXCL10-Hi TAM",
                  "Cycling TAM",
                  "MX1-Hi CD8+ T cell",
                  "Cycling Tumor",
                  "CLEC9A+ DC",
                  "Fibroblast",
                  "Mast cell"]


def filter_cell_types(adata, cell_types_to_remove=low_cell_types):
    bool_inds = ~adata.obs.MajorCluster.isin(cell_types_to_remove)
    print(f'removing {sum(~bool_inds)} cells')
    return adata[bool_inds, :]

# def filter_broad_data(adata):
#     ["CLEC9A + DC", "Cycling Tumor", "Fibroblast", "Mast cell"]


if __name__ == "__main__":
    adata = broad_quick_load()
    adata = filter_patients(adata)
    adata = filter_cell_types(adata)
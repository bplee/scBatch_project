import os
import sys
import numpy as np
import pandas as pd
import pyreadr
import rpy2.robjects as robjects
readRDS = robjects.r['readRDS']

if __name__ == "__main__":

    new_annot_file_path = '/data/leslie/krc3004/RCC_Alireza_Sep2020/ccRCC_6pat_cell_annotations_June2020.rds'

    new = readRDS(new_annot_file_path)

    column_names = new.colnames

    annot_data = pd.DataFrame(np.array(new).T)
    annot_data.columns = column_names
    # so this contains 31 labels instead of 16

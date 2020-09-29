import os
import sys
import numpy as np
import pandas as pd
import pyreadr
import rpy2.robjects as robjects
readRDS = robjects.r['readRDS']

WORKING_DIR = "/data/leslie/bplee/scBatch"
print("________CHANGING PATH_________")
sys.path.append(WORKING_DIR)
print("\tWorking dir appended to Sys path.")

if __name__ == "__main__":
    
    from ForBrennan.DIVA.dataset.rcc_loader_semi_sup import RccDatasetSemi

    # getting training and testing data
    TEST_PATIENT = 4
    X_DIM = 16323# 784 is the magic number for DIVA; 16323 is the max

    # getting training and testing data
    data_obj = RccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=True, test=True, diva=False)


    new_annot_file_path = '/data/leslie/krc3004/RCC_Alireza_Sep2020/ccRCC_6pat_cell_annotations_June2020.rds'
    new_batch_corrected_data = '/data/leslie/krc3004/RCC_Alireza_Sep2020/ccRCC_6pat_June2020.rds'

    raw_counts_file = './200929_raw_counts.rds'    

    new = readRDS(new_annot_file_path)

    column_names = new.columns

    # so this contains 31 labels instead of 16

    #data = readRDS(new_batch_corrected_data)

    

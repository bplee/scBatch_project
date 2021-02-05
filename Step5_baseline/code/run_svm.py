import sys
import os
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("CHANGING PATH")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

from Step0_Data.code,pkl_load_data import PdRccAllData


if __name__ == "__main__":
    data_obj = PdRccAllData()  # default args for this function will give me what I want
    raw_counts = data_obj.data.drop(['patient', 'cell_type'], axis=1)
    patients = data_obj.data.patient
    dim_out = len(np.unique(patients))
    y_onehot = np.eye(dim_out)[pd.factorize(patients)]
    cell_types = data_obj.data.cell_type

    svm = LinearSVC
    svm.fit(raw_counts, y_onehot)

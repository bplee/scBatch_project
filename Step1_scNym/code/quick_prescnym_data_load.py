from scnym.api import scnym_api
import torch
import os
import numpy as np
import anndata
import sys
import scanpy as sc
import pandas as pd
import scnym

import urllib
import json

# allow tensorboard outputs even though TF2 is installed
# broke the tensorboard/pytorch API
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# changing directory to project dir
print("________CHANGING WORKING DIR________")
print(os.getcwd())
WORKING_DIR = "/data/leslie/bplee/scBatch"
os.chdir(WORKING_DIR)
print("\tNew working dir: %s\n" % (os.getcwd()))

# adding the project dir to the path to import relevant modules below
print("________CHANGING PATH_________")
sys.path.append(WORKING_DIR)
print("\tWorking dir appended to Sys path.")

from ForBrennan.DIVA.dataset.rcc_loader_semi_sup import RccDatasetSemi

def get_Rcc_adata(test_patient, train_patient=None, x_dim=16323):
    """

    Parameters
    ----------
    test_patient: int
        integer in {0,..,5} (for old version of data)
    train_patient: int
        (default: None) int in {0,...,5} specifying which patient you want
        default is all

    Returns
    -------
    anndata.AnnData obj
        obj that contains all data information
    """
    # getting training and testing data
    TEST_PATIENT = test_patient
    X_DIM = x_dim # 784 is the magic number for DIVA; 16323 is the max

    # getting training and testing data
    data_obj = RccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=True, test=True, diva=False)

    cell_types = data_obj.cell_types

    # need to select one patient to use as training domain:
    TRAIN_PATIENT = train_patient #choose {0,...,5}

    if TRAIN_PATIENT is not None:
        # selecting all of the indices that mark our patient of interest
        train_patient_inds = data_obj.train_domain[:,TRAIN_PATIENT] == 1
        # using inds to select data for our patient
        train_patient_data = data_obj.train_data[train_patient_inds]
    else:
        train_patient_data = data_obj.train_data

    # making the data obj for our training and test patient
    train_adata = anndata.AnnData(np.array(train_patient_data))
    test_adata = anndata.AnnData(np.array(data_obj.test_data))

    # converting 1 hot vectors into int labels
    train_int_labels = np.array(data_obj.train_labels).dot(np.arange(len(data_obj.train_labels[0]), dtype=int))
    test_int_labels = np.array(data_obj.test_labels).dot(np.arange(len(data_obj.test_labels[0]), dtype=int))

    # creating vectors of cell labels (strings):
    train_cell_types = cell_types[train_int_labels]
    test_cell_types = cell_types[test_int_labels]

    # setting gold labels: (using names and not indices)
    train_adata.obs['cell_type'] = train_cell_types[train_patient_inds] # there are cell types for multiple patients so we index for the one we care about
    test_adata.obs['cell_type'] = test_cell_types

    # setting the semi_supervised labels:
    train_adata.obs['annotations'] = train_cell_types[train_patient_inds]
    test_adata.obs['annotations'] = 'Unlabeled'

    # concatenating data
    adata = train_adata.concatenate(test_adata)

    print("Returning adata and RccDatasetSemi loader obj")
    print("Test Patient: %d" % TEST_PATIENT)
    print("Train Patient: %d" % TRAIN_PATIENT)
    print("No. of Genes: %d" % X_DIM)

    return adata, data_obj

blurb = """
Loaded 1 useful annData object:
    - adata (ready as input to scnym, only 2 patients)

Loaded 1 useful class that holds all data:
    - data_obj (Rcc Semi super loader, all data)
Ready to train scnym
"""

if __name__ == "__main__":
    adata, data_obj = get_Rcc_adata(test_patient=5, train_patient=4)
    print(blurb)


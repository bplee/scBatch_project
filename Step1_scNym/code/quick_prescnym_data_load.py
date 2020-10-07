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

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

from ForBrennan.DIVA.dataset.rcc_loader_semi_sup import RccDatasetSemi
from Step0_Data.code.pkl_load_data import PdRccAllData


# this is not LOG NORMALIZED!


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
        obj that contains the following columns:
            - cell_type     (golden label cell type)
            - patient       (patient where cell came from)
            - annotations   (Same as `cell type` but all test points are 'Unlabeled')
            - batch         (boolean vector of training vs. test set)
    """
    # getting training and testing data
    TEST_PATIENT = test_patient
    X_DIM = x_dim # 784 is the magic number for DIVA; 16323 is the max

    # getting training and testing data
    data_obj = PdRccAllData()

    raw_counts = data_obj.data.drop(['patient', 'cell_type'], axis=1)
    patient_labels = data_obj.data.patient
    cell_labels = data_obj.data.cell_type

    patients = np.unique(data_obj.data.patient)
    cell_types = np.unique(data_obj.data.cell_type)

    # need to select one patient to use as training domain:
    TRAIN_PATIENT = train_patient  # choose {0,...,5}

    # selecting all of the indices that mark our testing patient
    test_patient_inds = patient_labels == patients[TEST_PATIENT]
    # using inds to select data for our patient
    test_patient_data = raw_counts[test_patient_inds]

    if TRAIN_PATIENT is None:
        train_patient_inds = ~test_patient_inds
        train_patient_data = raw_counts[train_patient_inds]
    else:
        # selecting all of the indices that mark our training patient
        train_patient_inds = patient_labels == patients[TRAIN_PATIENT]
        # using inds to select data for our patient
        train_patient_data = raw_counts[train_patient_inds]

    # making the data obj for our training and test patient
    train_adata = anndata.AnnData(np.array(train_patient_data))
    # train_adata = anndata.AnnData(np.array(train.train_data.reshape(train_cell_num, X_DIM)))
    test_adata = anndata.AnnData(np.array(test_patient_data))

    # setting gold labels: (using names and not indices)
    train_adata.obs['cell_type'] = np.array(cell_labels[train_patient_inds])  # there are cell types for multiple patients so we index for the one we care about
    test_adata.obs['cell_type'] = np.array(cell_labels[test_patient_inds])

    # setting the semi_supervised labels:
    train_adata.obs['annotations'] = np.array(cell_labels[train_patient_inds])
    test_adata.obs['annotations'] = 'Unlabeled'

    # concatenating data
    adata = train_adata.concatenate(test_adata)


    # getting training and testing data
    # data_obj = RccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=True, test=True, diva=False)
    #
    # patients = data_obj.patients
    # cell_types = data_obj.cell_types
    #
    # # need to select one patient to use as training domain:
    # TRAIN_PATIENT = train_patient #choose {0,...,5}
    #
    # # if TRAIN_PATIENT is not None:
    # #     # selecting all of the indices that mark our patient of interest
    # #     train_patient_inds = data_obj.train_domain[:,TRAIN_PATIENT] == 1
    # #     # using inds to select data for our patient
    # #     train_patient_data = data_obj.train_data[train_patient_inds]
    # # else:
    # #     train_patient_data = data_obj.train_data
    #
    # # making the data obj for our training and test patient
    # train_adata = anndata.AnnData(np.array(data_obj.train_data))
    # test_adata = anndata.AnnData(np.array(data_obj.test_data))
    #
    # # converting 1 hot patient vectors into ints
    # train_int_patients = np.array(data_obj.train_domain).dot(np.arange(len(data_obj.train_domain[0]))).astype(int)
    # test_int_patients = np.array(data_obj.test_domain).dot(np.arange(len(data_obj.test_domain[0]))).astype(int)
    # # creating vectors of patient names (strings):
    # train_patients = patients[train_int_patients]
    # test_patients = patients[test_int_patients]
    # # setting patient names: (using names and not indices)
    # train_adata.obs['patient'] = train_patients
    # test_adata.obs['patient'] = test_patients
    #
    # # converting 1 hot vectors into int labels (for cell types)
    # train_int_labels = np.array(data_obj.train_labels).dot(np.arange(len(data_obj.train_labels[0]), dtype=int))
    # test_int_labels = np.array(data_obj.test_labels).dot(np.arange(len(data_obj.test_labels[0]), dtype=int))
    # # creating vectors of cell types (strings):
    # train_cell_types = cell_types[train_int_labels]
    # test_cell_types = cell_types[test_int_labels]
    # # setting gold labels: (using names and not indices)
    # train_adata.obs['cell_type'] = train_cell_types
    # test_adata.obs['cell_type'] = test_cell_types
    #
    # # setting the semi_supervised labels:
    # train_adata.obs['annotations'] = train_cell_types
    # test_adata.obs['annotations'] = 'Unlabeled'
    #
    # if TRAIN_PATIENT is not None:
    #     train_adata = train_adata[train_adata.obs.patient == patients[TRAIN_PATIENT]]
    #
    # # concatenating data
    # adata = train_adata.concatenate(test_adata)

    print("Returning adata and RccDatasetSemi loader obj")
    print(f"Test Patient: {TEST_PATIENT}")
    if TRAIN_PATIENT is not None:
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
    adata, data_obj = get_Rcc_adata(test_patient=5)
    print(blurb)


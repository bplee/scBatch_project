from scnym.api import scnym_api
import torch
import os
import numpy as np
import pandas as pd
import anndata
import sys
import scnym
import scanpy as sc

import urllib
import json

from scvi.dataset import GeneExpressionDataset

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


def unwrap_list(lst):
    try:
        return np.array([row[0] for row in lst])
    except:
        return lst

def get_Rcc_adata(test_patient, train_patient=None, x_dim=16323, log_norm=True, shuffle=False):
    """

    Parameters
    ----------
    test_patient: int
        integer in {0,..,5} (for old version of data)
    train_patient: int, optional
        (default is None) int in {0,...,5} specifying which patient you want
        default is all
    x_dim : int, optional
        (default is 16323, max number of genes)
    log_norm : bool, optional
        (default is True) whether or not you want to log normalize the data
        (always should for scNym)

    Returns
    -------
    anndata.AnnData obj
        obj that contains the following columns:
            - cell_type     (golden label cell type)
            - patient       (patient where cell came from)
            - annotations   (Same as `cell type` but all test points are 'Unlabeled')
            - batch         (boolean vector of training vs. test set)
    """
    # 784 is the magic x_dim number for DIVA; 16323 is the max

    # getting training and testing data
    data_obj = PdRccAllData()

    raw_counts = data_obj.data.drop(['patient', 'cell_type'], axis=1)

    patient_labels = data_obj.data.patient
    cell_labels = data_obj.data.cell_type

    # patient_names = np.unique(data_obj.data.patient)
    # cell_type_names = np.unique(data_obj.data.cell_type)
    gene_names = raw_counts.columns

    raw_counts = np.array(raw_counts)
    if log_norm:
        print("Taking log norm of data")
        raw_counts = np.log(raw_counts + 1)

    # getting categorical indices for patients and cell_types
    patient_indices, patient_names = pd.factorize(data_obj.data.patient)
    cell_type_indices, cell_type_names = pd.factorize(data_obj.data.cell_type)

    # subsampling genes before selecting patient indices
    # TODO: I actually think I might want to do this in the other way, but the diva data load works in this way
    gene_dataset = GeneExpressionDataset()
    gene_dataset.populate_from_data(X=raw_counts,
                                    gene_names=gene_names,
                                    batch_indices=patient_indices,
                                    labels=cell_type_indices,
                                    remap_attributes=False)
    gene_dataset.subsample_genes(x_dim)

    adata = anndata.AnnData(gene_dataset.X)
    adata.obs['cell_type'] = np.array(cell_labels)
    adata.obs['batch'] = np.array(patient_labels)
    adata.obs['annotations'] = np.array(cell_labels)
    adata.obs['annotations'][patient_indices == test_patient] = 'Unlabeled'
    adata.obs['dist'] = 'train'
    adata.obs['train'][patient_indices == test_patient] = 'test'

    if train_patient is not None:
        keep_inds_bool = np.logical_or(patient_indices == train_patient, patient_indices == test_patient)
        keep_inds = np.arange(len(keep_inds_bool))[keep_inds_bool]
        adata = adata[keep_inds, :]


    ## OLD WAY
    # # selecting all of the indices that mark our testing patient
    # test_patient_inds = unwrap_list(gene_dataset.batch_indices == [test_patient])
    # # using inds to select data for our patient
    # test_patient_data = gene_dataset.X[test_patient_inds]
    #
    # # if there was no training patient
    # if train_patient is None:
    #     # training set will be all other patients
    #     train_patient_inds = ~test_patient_inds
    #     train_patient_data = gene_dataset.X[train_patient_inds]
    # else:
    #     # selecting all of the indices that mark our training patient
    #     train_patient_inds = unwrap_list(gene_dataset.batch_indices == [train_patient])
    #     # using inds to select data for our patient
    #     train_patient_data = gene_dataset.X[train_patient_inds]
    #
    # # making the data obj for our training and test patient
    # train_adata = anndata.AnnData(np.array(train_patient_data))
    # # train_adata = anndata.AnnData(np.array(train.train_data.reshape(train_cell_num, X_DIM)))
    # test_adata = anndata.AnnData(np.array(test_patient_data))
    #
    # # setting gold labels: (using names and not indices)
    # cell_labels = cell_type_names[unwrap_list(gene_dataset.labels)]
    #
    # train_adata.obs['cell_type'] = np.array(cell_labels[train_patient_inds])  # there are cell types for multiple patients so we index for the one we care about
    # test_adata.obs['cell_type'] = np.array(cell_labels[test_patient_inds])
    #
    # # setting the semi_supervised labels:
    # train_adata.obs['annotations'] = np.array(cell_labels[train_patient_inds])
    # test_adata.obs['annotations'] = 'Unlabeled'
    #
    # # concatenating data
    # adata = train_adata.concatenate(test_adata)
    # if train_patient == None:
    #     adata.obs['batch'] = np.array(data_obj.data.patient)
    #     print(f"Setting batch indices for all patient training")

    # shuffle data:
    if shuffle == True:
        inds = np.arange(len(adata))
        np.random.shuffle(inds)
        adata.obs.batch = np.array(adata.obs.batch[inds])
        adata.obs.annotations = np.array(adata.obs.annotations[inds])
        adata.obs.cell_type = np.array(adata.obs.cell_type[inds])
        adata.X = np.array(adata.X[inds])

    print("Returning adata and RccDatasetSemi loader obj")
    print(f"Test Patient: {test_patient}")
    if train_patient is not None:
        print(f"Train Patient: {train_patient}")
    print(f"No. of Genes: {x_dim}")

    return adata, data_obj

blurb = """
Loaded 1 useful annData object:
    - adata (ready as input to scnym, only 2 patients)

Loaded 1 useful class that holds all data:
    - data_obj (Rcc Semi super loader, all data)
Ready to train scnym
"""

if __name__ == "__main__":
    adata, data_obj = get_Rcc_adata(test_patient=5, train_patient=4, x_dim=784)
    print(blurb)


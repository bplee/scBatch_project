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
print(os.getcwd())
WORKING_DIR = "/data/leslie/bplee/scBatch"

# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

# from ForBrennan.DIVA.dataset.rcc_loader_semi_sup import RccDatasetSemi
from Step0_Data.code.pkl_load_data import PdRccAllData

# this is not LOG NORMALIZED!

# getting training and testing data
TEST_PATIENT = 4
X_DIM = 10000# 784 is the magic number for DIVA

# getting training and testing data
# train = RccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=True, diva=False)
# test = RccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=False, diva=False)
data_obj = PdRccAllData()

raw_counts = data_obj.data.drop(['patient', 'cell_type'], axis=1)
patient_labels = data_obj.data.patient
cell_labels = data_obj.data.cell_type

patients = np.unique(data_obj.data.patient)
cell_types = np.unique(data_obj.data.cell_type)

# need to select one patient to use as training domain:
TRAIN_PATIENT = 3 #choose {0,...,5}

# selecting all of the indices that mark our training patient
train_patient_inds = patient_labels == patients[TRAIN_PATIENT]
# using inds to select data for our patient
train_patient_data = raw_counts[train_patient_inds]

# selecting all of the indices that mark our testing patient
test_patient_inds = patient_labels == patients[TEST_PATIENT]
# using inds to select data for our patient
test_patient_data = raw_counts[test_patient_inds]


# making the data obj for our training and test patient
train_adata = anndata.AnnData(np.array(train_patient_data))
# train_adata = anndata.AnnData(np.array(train.train_data.reshape(train_cell_num, X_DIM)))
test_adata = anndata.AnnData(np.array(test_patient_data))

# setting gold labels: (using names and not indices)
train_adata.obs['cell_type'] = cell_labels[train_patient_inds] # there are cell types for multiple patients so we index for the one we care about
test_adata.obs['cell_type'] = cell_labels[test_patient_inds]

# setting the semi_supervised labels:
train_adata.obs['annotations'] = cell_labels[train_patient_inds]
test_adata.obs['annotations'] = 'Unlabeled'


# concatenating data
#adata = train_adata.concatenate(test_adata)

# to balance the test distribution
def get_balanced_classes(adata):
    """
    Returns an anndata obj with balanced labels

    Input:
        anndata obj with stuff in adata.obs['cell_type']

    Returns:
        Smaller anndata obj with balanced labels
    
    """
    counts = adata.obs.cell_type.value_counts()
    min_cell_type, min_num = counts.index[-1], counts[-1]
    rtn = []
    for i, cell_type_count in enumerate(counts):
        cell_type = counts.index[i]
        cell_type_inds = np.array([i for i,val in enumerate(adata.obs.cell_type == cell_type) if val]) # this line is returning the inds of all points with given cell type
        a = np.random.choice(cell_type_count, min_num, replace=False) # choose n points from the list
        n_inds = cell_type_inds[a] # sampled n random indices
        rtn.extend(n_inds)
    return adata[rtn,:]

# TODO: shouldnt do this re naming, should just change the test/train patients
balanced_train_adata = get_balanced_classes(test_adata)
print("created balanced_train_adata from test_adata")

balanced_test_adata = get_balanced_classes(train_adata)
print("created balanced_test_adata from train_adata")


#fixing annotations because we switched the train and test sets
balanced_train_adata.obs.annotations = balanced_train_adata.obs.cell_type
balanced_test_adata.obs.annotations = "Unlabeled"

adata = balanced_train_adata.concatenate(balanced_test_adata)

# training scnym
scnym_api(adata=adata,
          task='train',
          config='no_new_identity',
          out_path='./scnym_test_output',  # this is going in WORKING DIR
          groupby='annotations')
print("Done training. Now for prediction")
scnym_api(
    adata=adata,
    task='predict',
    key_added='scNym',
    config='no_new_identity',
    trained_model='./scnym_test_output'
)

sc.pp.neighbors(adata, use_rep='X_scnym', n_neighbors=30)
sc.tl.umap(adata, min_dist=.3)
# the following are the scnym internal embeddings colored by batch and cell type
sc.pl.umap(adata, color='batch', size=5, alpha=.2, save='scnym_embedding_batch.png') 
sc.pl.umap(adata, color='cell_type', size=5, alpha=.2, save='scnym_embedding_celltypes.png')

sc.pp.neighbors(adata, use_rep='X', n_neighbors=30)
sc.tl.umap(adata, min_dist=.3)
sc.pl.umap(adata, color='cell_type', size=5, alpha=.2, save='scnym_og_data_umap_celltype.png')

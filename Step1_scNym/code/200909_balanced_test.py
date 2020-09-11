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

# getting training and testing data
TEST_PATIENT = 4
X_DIM = 10000# 784 is the magic number for DIVA

# getting training and testing data
train = RccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=True, diva=False)
test = RccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=False, diva=False)

cell_types = train.cell_types

# need to select one patient to use as training domain:
TRAIN_PATIENT = 3 #choose {0,...,5}

# selecting all of the indices that mark our patient
train_patient_inds = train.train_domain[:,TRAIN_PATIENT] == 1 
# using inds to select data for our patient
train_patient_data = train.train_data[train_patient_inds] #doing log transformation here

# making the data obj for our training and test patient
train_adata = anndata.AnnData(np.array(train_patient_data))
# train_adata = anndata.AnnData(np.array(train.train_data.reshape(train_cell_num, X_DIM)))
test_adata = anndata.AnnData(np.array(test.test_data))

# converting 1 hot vectors into int labels
train_int_labels = np.array(train.train_labels).dot(np.arange(len(train.train_labels[0]), dtype=int))
test_int_labels = np.array(test.test_labels).dot(np.arange(len(test.test_labels[0]), dtype=int))

# creating vectors of cell labels:
train_cell_types = train.cell_types[train_int_labels]
test_cell_types = test.cell_types[test_int_labels]

# setting gold labels: (using names and not indices)
train_adata.obs['cell_type'] = train_cell_types[train_patient_inds] # there are cell types for multiple patients so we index for the one we care about
test_adata.obs['cell_type'] = test_cell_types

# setting the semi_supervised labels:
train_adata.obs['annotations'] = train_cell_types[train_patient_inds]
test_adata.obs['annotations'] = 'Unlabeled'

print('hey')
print(type(train_int_labels))
print(train_int_labels.dtype)

# concatenating data
adata = train_adata.concatenate(test_adata)

# debugging lines:
print(adata.obs)


blurb = """
Loaded 3 useful annData objects:
	- adata (ready as input to scnym)
	- train_adata
	- test_adata
Ready to train scnym
"""
print(blurb)

# now we want to create a balanced train/testing set to check the data

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


get_balanced_classes(train_adata)



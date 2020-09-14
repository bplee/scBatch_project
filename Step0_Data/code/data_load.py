from scnym.api import scnym_api
import torch
import os
import numpy as np
import anndata
import sys
import scanpy as sc
import pandas as pd
import scnym

import matplotlib.pyplot as plt

import urllib
import json
from sklearn.decomposition import PCA

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
X_DIM = 16323# 784 is the magic number for DIVA; 16323 is the max

# getting training and testing data
data_obj = RccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=True, test=True, diva=False)

patients = data_obj.patients
cell_types = data_obj.cell_types

# making the data obj for our training and test patient
train_adata = anndata.AnnData(np.array(data_obj.train_data))
# train_adata = anndata.AnnData(np.array(train.train_data.reshape(train_cell_num, X_DIM)))
test_adata = anndata.AnnData(np.array(data_obj.test_data))

# converting 1 hot vectors into int labels
train_labels_int = np.array(data_obj.train_labels).dot(np.arange(len(data_obj.train_labels[0]), dtype=int))
test_labels_int = np.array(data_obj.test_labels).dot(np.arange(len(data_obj.test_labels[0]), dtype=int))

# creating vectors of cell labels (strings):
train_cell_types = cell_types[train_labels_int]
test_cell_types = cell_types[test_labels_int]

# setting gold labels: (using names and not indices)
train_adata.obs['cell_type'] = train_cell_types # all training patients included
test_adata.obs['cell_type'] = test_cell_types

# setting the semi_supervised labels:
train_adata.obs['annotations'] = train_cell_types
test_adata.obs['annotations'] = 'Unlabeled'

# setting a column of patients with each value as a str name
train_adata.obs['patient'] = np.array(data_obj.train_domain).dot(np.arange(len(data_obj.train_domain[0]),dtype=int))
test_adata.obs['patient'] = np.array(data_obj.test_domain).dot(np.arange(len(data_obj.test_domain[0]), dtype=int))

print('hey')
# print(train_int_labels.dtype)

# concatenating data
adata = train_adata.concatenate(test_adata)

# debugging lines:
print(adata.obs)

blurb = """
Loaded 3 useful annData objects:
        - adata (all data loaded, all pateints)
        - train_adata (train patient)
        - test_adata (test patient)
Loaded 1 useful class that holds all data:
        - data_obj (Rcc Semi super loader, all data)
"""
print(blurb)


def scatter_color(x, y, groups, savepath='./'):
    """
    Saves a figure via matplotlib
    Figure will be colored by integer labels of groups
    Figure will contain legend of the group names
    """
    groups = np.array(groups)
    for group in np.unique(groups):
        coor = np.array([[x[i], y[i]] for i in range(len(x)) if groups[i] == group])
        plt.scatter(coor[:,0], coor[:,1], alpha=.4, s=6, label=group)
    plt.legend()
    plt.savefig(savepath)
    print("Done")
    print("Saved fig to %s" % savepath)

pca_obj = PCA(n_components=2)
proj = pca_obj.fit_transform(adata.X)

scatter_color(proj[:,0], proj[:,1], adata.obs.patient, savepath='/data/leslie/bplee/scBatch/Step0_Data/figs/pca_all_patients.png')

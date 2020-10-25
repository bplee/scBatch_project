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
X_DIM = 16323# 784 is the magic number for DIVA; 16323 is the max

# getting training and testing data
train = RccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=True, diva=False)
test = RccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=False, diva=False, test=True)

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
2 Useful data objs:
	- train
	- test
Ready to train scnym
"""
print(blurb)
print("Test patient: %d" %TEST_PATIENT)
print("Train patient: %d" %TRAIN_PATIENT)

# THE FOLLOWING IS CODE FROM new_prescnym_data_load.py THAT I PROBABLY WON'T USE ANY MORE
# BUT DIDN'T WANT TO DITCH

# getting training and testing data
# data_obj = RccDatasetSemi(test_patient=test_patient, x_dim=X_DIM, train=True, test=True, diva=False)
#
# patients = data_obj.patients
# cell_types = data_obj.cell_types
#
# # need to select one patient to use as training domain:
# train_patient = train_patient #choose {0,...,5}
#
# # if train_patient is not None:
# #     # selecting all of the indices that mark our patient of interest
# #     train_patient_inds = data_obj.train_domain[:,train_patient] == 1
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
# if train_patient is not None:
#     train_adata = train_adata[train_adata.obs.patient == patients[train_patient]]
#
# # concatenating data
# adata = train_adata.concatenate(test_adata)

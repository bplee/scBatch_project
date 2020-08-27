from scnym.api import scnym_api
import torch
import os
import numpy as np
import anndata
import sys
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
TEST_PATIENT = 5
X_DIM = 784

# getting training and testing data
train = RccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=True, diva=False)
test = RccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=False, diva=False)

# dont need this anymore
# reshaping it from its weird diva structure
#train_cell_num = train.train_data.shape[0]
#test_cell_num = test.test_data.shape[0]

# need to select one patient to use as training domain:
TRAIN_PATIENT = 3 #choose {0,...,5}

# selecting all of the indices that mark our patient
train_patient_inds = train.train_domain[:,TRAIN_PATIENT] == 1 
# using inds to select data for our patient
train_patient_data = train.train_data[train_patient_inds]

print('run_scnym.py line 54')
print("train_patient_data.shape", train_patient_data.shape)
print("test_data.shape", test_data.shape)

# making the data obj for our training and test patient
train_adata = anndata.AnnData(np.array(train_patient_data))
# train_adata = anndata.AnnData(np.array(train.train_data.reshape(train_cell_num, X_DIM)))
test_adata = anndata.AnnData(np.array(test.test_data))

# converting 1 hot vectors into int labels
train_int_labels = np.array(train.train_labels).dot(np.arange(len(train.train_labels[0]), dtype=float))

# setting labels by grabbing the labels for our training partient
train_adata.obs['annotations'] = train_int_labels[train_patient_inds]
test_adata.obs['annotations'] = 'Unlabeled'


print('hey')
print(type(train_int_labels))
print(train_int_labels.dtype)

# concatenating data
adata = train_adata.concatenate(test_adata)

# debugging lines:
print(adata.obs)

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

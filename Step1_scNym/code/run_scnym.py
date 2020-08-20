from scnym.api import scnym_api
import torch
import os
import numpy as np
import anndata

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
train = RccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=True)
test = RccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=False)

# reshaping it from its weird diva structure
train_cell_num = train.train_data.shape[0]
test_cell_num = test.test_data.shape[0]

train_adata = anndata.AnnData(np.array(train.train_data.reshape(train_cell_num, X_DIM)))
test_adata = anndata.AnnData(np.array(test.test_data.reshape(test_cell_num, X_DIM)))

# converting 1 hot vectors into int labels
train_int_labels = np.array(train.train_labels).dot(np.arrange(len(train.train_labels[0])))

# setting labels
train_adata.obs['annotations'] = train_int_labels
test_adata.obs['annotations'] = 'Unlabeled'

# concatenating data
adata = train_adata.concatenate(test_adata)

# training scnym
scnym_api(adata=adata,
          task='train',
          config='no_new_identity',
          out_path='./scnym_test_output',  # this is going in WORKING DIR
          groupby='annotations')


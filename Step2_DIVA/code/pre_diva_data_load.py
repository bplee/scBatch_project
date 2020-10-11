import os
import sys
import argparse
import time
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np


WORKING_DIR = "/data/leslie/bplee/scBatch"

# adding the project dir to the path to import relevant modules below
print("________CHANGING PATH_________")
sys.path.append(WORKING_DIR)
print("\tWorking dir appended to Sys path.")

from ForBrennan.DIVA.model.model_diva import DIVA
from DIVA.dataset.rcc_loader_semi_sup import RccDatasetSemi
from Step0_Data.code.new_data_load import NewRccDatasetSemi

test_patient =5
batch_size=100



train_loader_sup = RccDatasetSemi(test_patient,
                 train=True, x_dim=784)
train_loader_unsup = RccDatasetSemi(test_patient,
                 train=False, x_dim=784)

new_train_loader_sup = NewRccDatasetSemi(test_patient,
                 train=True, x_dim=784)
new_train_loader_unsup = NewRccDatasetSemi(test_patient,
                 train=False, x_dim=784)

data_loaders = {}
data_loaders['sup'], data_loaders['unsup'] = data_utils.DataLoader(train_loader_sup, batch_size=1), data_utils.DataLoader(train_loader_unsup, batch_size=1)

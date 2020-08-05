import os
import sys

import argparse
import time
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils



# changing directory to project dir

print(os.getcwd())
WORKING_DIR = "/data/leslie/bplee/scBatch"
os.chdir(WORKING_DIR)
print("New working dir: %s\n" % (os.getcwd()))

# adding the project dir to the path to import relevant modules
sys.path.append(WORKING_DIR)
print("Sys path appended to")

# importing a data class
from DIVA.model.model_diva import DIVA
from DIVA.dataset.rcc_loader_semi_sup import RccDatasetSemi

print("trying to use the class")
data_obj = RccDatasetSemi(test_patient=1, x_dim=200)

print("done")
print(type(data_obt))


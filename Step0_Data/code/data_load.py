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
print("________CHANGING WORKING DIR________")
print(os.getcwd())
WORKING_DIR = "/data/leslie/bplee/scBatch"
os.chdir(WORKING_DIR)
print("\tNew working dir: %s\n" % (os.getcwd()))

# adding the project dir to the path to import relevant modules
print("________CHANGING PATH_________")
sys.path.append(WORKING_DIR)
print("\tWorking dir appened to Sys path.")

# importing a data class
from ForBrennan.DIVA.model.model_diva import DIVA
from ForBrennan.DIVA.dataset.rcc_loader_semi_sup import RccDatasetSemi

print("trying to use the class")
data_obj = RccDatasetSemi(test_patient=1, x_dim=200)

print("done")
print(type(data_obt))


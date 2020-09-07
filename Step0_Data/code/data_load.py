import os
import sys

import argparse
import time
import numpy as np
import torch


# changing directory to project dir
print("________CHANGING WORKING DIR________")
print(os.getcwd())
WORKING_DIR = "/data/leslie/bplee/scBatch"
os.chdir(WORKING_DIR)
print("\tNew working dir: %s\n" % (os.getcwd()))

# adding the project dir to the path to import relevant modules
print("________CHANGING PATH_________")
sys.path.append(WORKING_DIR)
print("\tWorking dir appended to Sys path.")

# importing a data class
from ForBrennan.DIVA.model.model_diva import DIVA
from ForBrennan.DIVA.dataset.rcc_loader_semi_sup import RccDatasetSemi

print("trying to use the class")
#X_DIM = 784
X_DIM = 1200
data_obj = RccDatasetSemi(test_patient=1, x_dim=1200, train=True, diva=False)

print("done")
#Data is now stored in data_obj.train_data etc



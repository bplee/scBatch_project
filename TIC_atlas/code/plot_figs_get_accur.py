import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import anndata
import scanpy as sc
import torch.utils.data as data_utils

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

from Step0_Data.code.starter import ensure_dir
from TIC_atlas.code.load_data import load_TIC_diva_datasets
from Step6_RCC_to_CRC.code.rcc_to_crc_diva import get_accuracy

d_accuracies = []
y_accuracies = []
weighted_y_accuracies = []


for i in range(10):
    model_name = f"210323_TIC_no_conv_test_pat_[{i}]"
    print(model_name)
    model = torch.load(model_name + ".model")
    args = torch.load(model_name + ".config")
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}
    train_loader, test_loader = load_TIC_diva_datasets(test_domain=args.test_patient, train_domain=args.train_patient)
    data_loaders = {}
    data_loaders['sup'] = data_utils.DataLoader(train_loader, batch_size=args.batch_size, shuffle=False)
    data_loaders['unsup'] = data_utils.DataLoader(test_loader, batch_size=args.batch_size, shuffle=False)
    ensure_dir("./cm_figs")
    a = get_accuracy(data_loaders['unsup'], model, device, save=model_name)
    d_accuracies.append(a[0])
    y_accuracies.append(a[1])
    weighted_y_accuracies.append(a[2])



print(f"d_accur:{d_accuracies}")
print(f"d_accur:{y_accuracies}")
print(f"d_accur:{weighted_y_accuracies}")

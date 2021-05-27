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

from Step0_Data.code.starter import *
from TIC_atlas.code.load_data import load_patient_TIC_diva_datasets
from Step6_RCC_to_CRC.code.rcc_to_crc_diva import get_accuracy

d_accuracies = np.zeros((10,10))
y_accuracies = np.zeros((10,10))
weighted_y_accuracies = np.zeros((10,10))


for test_pat in range(0, 10):
    y_accur = []
    w_y_accur = []
    d_accur = []
    for train_pat in range(test_pat+1, 10):
        model_name = f"210329_TIC_no_conv_test_pat_[{test_pat}]_train_pat_[{train_pat}]"
        print(model_name)
        if not os.path.exists(model_name + ".model") or not os.path.exists(model_name + ".config"):
            print(f"skipping {model_name}")
            continue
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
        d_accuracies[test_pat, train_pat] = a[0]
        y_accuracies[test_pat, train_pat] = a[1]
        weighted_y_accuracies[train_pat, test_pat] = a[2]
    print(f"test_patient: {test_pat}")
    print(f" d_accur: {d_accuracies[test_pat]}")
    print(f" y_accur: {y_accuracies[test_pat]}")
    print(f" w_y_accur: {weighted_y_accuracies[test_pat]}")

print(f"d_accur:{d_accuracies}")
print(f"d_accur:{y_accuracies}")
print(f"d_accur:{weighted_y_accuracies}")





y_accur = []
w_y_accur = []
d_accur = []
train_pat = None
for test_pat in range(55,61):
    model_name = f"210329_TIC_no_conv_test_pat_[{test_pat}]_train_pat_{train_pat}"
    print(model_name)
    if not os.path.exists(model_name + ".model") or not os.path.exists(model_name + ".config"):
        print(f"skipping {model_name}")
        continue
    model = torch.load(model_name + ".model")
    args = torch.load(model_name + ".config")
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}
    train_loader, test_loader = load_patient_TIC_diva_datasets(test_domain=args.test_patient, train_domain=args.train_patient)
    data_loaders = {}
    data_loaders['sup'] = data_utils.DataLoader(train_loader, batch_size=args.batch_size, shuffle=False)
    data_loaders['unsup'] = data_utils.DataLoader(test_loader, batch_size=args.batch_size, shuffle=False)
    ensure_dir("./cm_figs")
    a = get_accuracy(data_loaders['unsup'], model, device, save=model_name)
    print(a)
    d_accur.append(a[0])
    y_accur.append(a[1])
    w_y_accur.append(a[2])
print(f"test_patient: {test_pat}")
print(f" d_accur: {d_accuracies[test_pat]}")
print(f" y_accur: {y_accuracies[test_pat]}")
print(f" w_y_accur: {weighted_y_accuracies[test_pat]}")

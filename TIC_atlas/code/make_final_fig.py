# this one is to get the correct figure for zy where each domain gets plotted individually

import pandas as pd
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
import seaborn as sns

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

from ForBrennan.DIVA.model.model_diva_no_convolutions import DIVA
from Step6_RCC_to_CRC.code.rcc_to_crc_test import *
import anndata
import torch
import torch.utils.data as data_utils
from scvi.dataset import GeneExpressionDataset
import scanpy as sc
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from Step6_RCC_to_CRC.code.rcc_to_crc_diva import *
from TIC_atlas.code.load_data import *

for test_pat in range(10):
    model_name = f"210323_TIC_no_conv_test_pat_[{test_pat}]"
    print(model_name)
    model = torch.load(model_name + ".model")
    args = torch.load(model_name + ".config")
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}
    train_loader, test_loader = load_TIC_diva_datasets(test_domain=args.test_patient,
                                                       train_domain=args.train_patient)
    data_loaders = {}
    data_loaders['sup'] = data_utils.DataLoader(train_loader, batch_size=args.batch_size, shuffle=False)
    data_loaders['unsup'] = data_utils.DataLoader(test_loader, batch_size=args.batch_size, shuffle=False)
    ensure_dir("./cm_figs")
    cell_types = test_loader.cell_types
    patients = test_loader.patients
    # trying to plot training data
    actuals_d, actuals_y, zy_, zd_, zx_ = [], [], [], [], []
    with torch.no_grad():
        # Train
        # patients_train = np.delete(patients, test_patient)
        i = 0
        for (xs, ys, ds) in data_loaders['sup']:
            i = i + 1
            # To device
            xs, ys, ds = xs.to(device), np.array(ys), np.array(ds)
            # use classification function to compute all predictions for each batch
            zy_loc, zy_scale = model.qzy(xs)
            zd_loc, zd_scale = model.qzd(xs)
            zx_loc, zx_scale = model.qzx(xs)
            zx_.append(np.array(zx_loc.cpu()))
            zy_.append(np.array(zy_loc.cpu()))
            zd_.append(np.array(zd_loc.cpu()))
            # getting integer labels here
            actuals_d.append(np.argmax(ds, axis=1))
            actuals_y.append(np.argmax(ys, axis=1))
            # if i == 50:
            if i == len(data_loaders['sup']):
                break
        zy = np.vstack(zy_)
        zd = np.vstack(zd_)
        zx = np.vstack(zx_)
        labels_y = np.hstack(actuals_y)
        labels_d = np.hstack(actuals_d)
        zy_adata, zd_adata, zx_adata = [anndata.AnnData(_) for _ in [zy, zd, zx]]
        adatas = [zy_adata, zd_adata, zx_adata]
        name = ['zy', 'zd', 'zx']
        zy_adata.obs['batch'] = patients[labels_d]
        zd_adata.obs['cell_type'] = cell_types[labels_y]
        train_cell_type_encoding = zy_adata
        train_batch_encoding = zd_adata
        # for i, _ in enumerate(adatas):
        #     _.obs['batch'] = patients[labels_d]
        #     _.obs['cell_type'] = cell_types[labels_y]
        #     save_name = f"_{fig_name}_train_set_{name[i]}.png"
        #     sc.pp.neighbors(_, use_rep="X", n_neighbors=15)
        #     sc.tl.umap(_, min_dist=.3)
        #     sc.pl.umap(_, color=['batch', 'cell_type'], save=save_name)
    actuals_d, actuals_y, zy_, zd_, zx_ = [], [], [], [], []
    with torch.no_grad():
        # test
        # patients_train = np.delete(patients, test_patient)
        i = 0
        for (xs, ys, ds) in data_loaders['unsup']:
            i = i + 1
            # To device
            xs, ys, ds = xs.to(device), np.array(ys), np.array(ds)
            # use classification function to compute all predictions for each batch
            zy_loc, zy_scale = model.qzy(xs)
            zd_loc, zd_scale = model.qzd(xs)
            zx_loc, zx_scale = model.qzx(xs)
            zx_.append(np.array(zx_loc.cpu()))
            zy_.append(np.array(zy_loc.cpu()))
            zd_.append(np.array(zd_loc.cpu()))
            # getting integer labels here
            actuals_d.append(np.argmax(ds, axis=1))
            actuals_y.append(np.argmax(ys, axis=1))
            # if i == 50:
            if i == len(data_loaders['unsup']):
                break
        zy = np.vstack(zy_)
        zd = np.vstack(zd_)
        zx = np.vstack(zx_)
        labels_y = np.hstack(actuals_y)
        labels_d = np.hstack(actuals_d)
        zy_adata, zd_adata, zx_adata = [anndata.AnnData(_) for _ in [zy, zd, zx]]
        adatas = [zy_adata, zd_adata, zx_adata]
        name = ['zy', 'zd', 'zx']
        zy_adata.obs['batch'] = patients[labels_d]
        zd_adata.obs['cell_type'] = cell_types[labels_y]
        test_cell_type_encoding = zy_adata
        test_batch_encoding = zd_adata
        # for i, _ in enumerate(adatas):
        #     _.obs['batch'] = patients[labels_d]
        #     _.obs['cell_type'] = cell_types[labels_y]
        #     save_name = f"_{fig_name}_test_set_{name[i]}.png"
        #     sc.pp.neighbors(_, use_rep="X", n_neighbors=15)
        #     sc.tl.umap(_, min_dist=.3)
        #     sc.pl.umap(_, color=['batch', 'cell_type'], save=save_name)
    full_zy = train_cell_type_encoding.concatenate(test_cell_type_encoding)
    full_zd = train_batch_encoding.concatenate(test_batch_encoding)
    sc.pp.neighbors(full_zy, n_neighbors=15)
    sc.pp.neighbors(full_zd, n_neighbors=15)
    sc.tl.umap(full_zy, min_dist=.3)
    sc.tl.umap(full_zd, min_dist=.3)
    sc.pl.umap(full_zy, color=['batch', 'cell_type'], save=f"_{fig_name}_train+test_zy.png")
    sc.pl.umap(full_zd, color=['batch', 'cell_type'], save=f"_{fig_name}_train+test_zd.png")
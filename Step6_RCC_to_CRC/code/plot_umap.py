import os
import sys

import argparse

import torch.utils.data as data_utils

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import anndata
import scanpy as sc

WORKING_DIR = "/data/leslie/bplee/scBatch"

# adding the project dir to the path to import relevant modules below
print("________CHANGING PATH_________")
sys.path.append(WORKING_DIR)
print("\tWorking dir appended to Sys path.")
#from paper_experiments.rotated_mnist.dataset.rcc_loader import RccDataset
from DIVA.dataset.rcc_loader_semi_sup import RccDatasetSemi
from Step0_Data.code.new_data_load import NewRccDatasetSemi as RccDatasetSemi
from Step0_Data.code.starter import get_valid_diva_models
from Step6_RCC_to_CRC.rcc_to_crc_test import *

def plot_umap(train_loader, test_loader, model, cell_types, patients, model_name, device, empty_zx=False):
    model.eval()
    """
    get the latent factors and plot the UMAP plots
    produces 18 plots/testpatient: [zy, zd, zx] x [patient label, cell type label] x [train, test, train+test]
    """
    actuals_d, actuals_y, zy_, zd_, zx_ = [], [], [], [], []

    with torch.no_grad():
        # Train
        # patients_train = np.delete(patients, test_patient)
        i = 0
        for (xs, ys, ds) in train_loader:
            i = i + 1
            # To device
            xs, ys, ds = xs.to(device), np.array(ys), np.array(ds)

            # use classification function to compute all predictions for each batch
            zy_loc, zy_scale = model.qzy(xs)
            zd_loc, zd_scale = model.qzd(xs)
            if not empty_zx:
                zx_loc, zx_scale = model.qzx(xs)
                zx_.append(np.array(zx_loc.cpu()))
            zy_.append(np.array(zy_loc.cpu()))
            zd_.append(np.array(zd_loc.cpu()))
            actuals_d.append(np.argmax(ds,axis=1))
            actuals_y.append(np.argmax(ys,axis=1))
            if i == 50:
               break

        zy = torch.cat(zy_)
        zd = torch.cat(zd_)
        if not empty_zx:
            zx = torch.cat(zx_)
        labels_y = torch.cat(actuals_y)
        labels_d = torch.cat(actuals_d)


        if not empty_zx:
            zy_adata, zd_adata, zx_adata = [anndata.AnnData(_) for _ in [zy, zd, zx]]
            adatas = [zy_adata, zd_adata, zx_adata]
        else:
            zy_adata, zd_adata = [anndata.AnnData(_) for _ in [zy, zd]]
            adatas = [zy_adata, zd_adata]

        name = ['zy', 'zd', 'zx']

        for i, _ in enumerate(adatas):
            _.obs['batch'] = patients[labels_d]
            _.obs['cell_type'] = cell_types[labels_y]
            save_name = f"_{model_name}_train_set_{name[i]}.png"

            sc.pp.neighbors(_, use_rep="X", n_neighbors=15)
            sc.tl.umap(_, min_dist=.3)
            sc.pl.umap(_, color=['batch', 'cell_type'], size=15, alpha=.8, save=save_name)
            # sc.pl.umap(_, color='cell_type', size=15, alpha=.8, save=save_name_cell_type)


        ## Test

        actuals_d, actuals_y, zy_, zd_, zx_ = [], [], [], [], []
        i = 0
        for (xs, ys, ds) in test_loader:
            i = i + 1
            # To device
            xs, ys, ds = xs.to(device), np.array(ys), np.array(ds)

            # use classification function to compute all predictions for each batch
            zy_loc, zy_scale = model.qzy(xs)
            zd_loc, zd_scale = model.qzd(xs)
            if not empty_zx:
                zx_loc, zx_scale = model.qzx(xs)
                zx_.append(np.array(zx_loc.cpu()))
            zy_.append(np.array(zy_loc.cpu()))
            zd_.append(np.array(zd_loc.cpu()))
            actuals_d.append(np.argmax(ds, axis=1))
            actuals_y.append(np.argmax(ys, axis=1))
            if i == 50:
                break

        zy = torch.cat(zy_)
        zd = torch.cat(zd_)
        if not empty_zx:
            zx = torch.cat(zx_)
        labels_y = torch.cat(actuals_y)
        labels_d = torch.cat(actuals_d)

        if not empty_zx:
            zy_adata, zd_adata, zx_adata = [anndata.AnnData(_) for _ in [zy, zd, zx]]
            adatas = [zy_adata, zd_adata, zx_adata]
        else:
            zy_adata, zd_adata = [anndata.AnnData(_) for _ in [zy, zd]]
            adatas = [zy_adata, zd_adata]

        name = ['zy', 'zd', 'zx']
        for i, _ in enumerate(adatas):
            _.obs['batch'] = patients[labels_d]
            _.obs['cell_type'] = cell_types[labels_y]
            # save_name_pat = '_diva_new_semi_sup_train_' + name[i] + '_by_batches_heldout_pat_' + str(test_patient) + '.png'
            # save_name_cell_type = '_diva_new_semi_sup_train_' + name[i] + '_by_label_heldout_pat_' + str(test_patient) + '.png'
            save_name = f"_{model_name}_test_set_{name[i]}.png"

            sc.pp.neighbors(_, use_rep="X", n_neighbors=15)
            sc.tl.umap(_, min_dist=.3)
            sc.pl.umap(_, color=['batch', 'cell_type'], size=15, alpha=.8, save=save_name)

        ## Train + Test

        # patients = np.append(patients_train, patients[test_patient])
        actuals_d, actuals_y, zy_, zd_, zx_ = [], [], [], [], []
        i = 0
        for (xs, ys, ds) in train_loader:
            i = i + 1
            # To device
            xs, ys, ds = xs.to(device), np.array(ys), np.array(ds)

            # use classification function to compute all predictions for each batch
            zy_loc, zy_scale = model.qzy(xs)
            zd_loc, zd_scale = model.qzd(xs)
            if not empty_zx:
                zx_loc, zx_scale = model.qzx(xs)
                zx_.append(np.array(zx_loc.cpu()))
            zy_.append(np.array(zy_loc.cpu()))
            zd_.append(np.array(zd_loc.cpu()))
            actuals_d.append(np.argmax(ds, axis=1))
            actuals_y.append(np.argmax(ys, axis=1))
            if i == 50:
                break

        i = 0
        for (xs, ys, ds) in test_loader:
            i = i + 1
            # To device
            xs, ys, ds = xs.to(device), np.array(ys), np.array(ds)

            # use classification function to compute all predictions for each batch
            zy_loc, zy_scale = model.qzy(xs)
            zd_loc, zd_scale = model.qzd(xs)
            if not empty_zx:
                zx_loc, zx_scale = model.qzx(xs)
                zx_.append(np.array(zx_loc.cpu()))
            zy_.append(np.array(zy_loc.cpu()))
            zd_.append(np.array(zd_loc.cpu()))
            actuals_d.append(np.argmax(ds, axis=1))
            actuals_y.append(np.argmax(ys, axis=1))
            if i == 10:
                break

        zy = torch.cat(zy_)
        zd = torch.cat(zd_)
        if not empty_zx:
            zx = torch.cat(zx_)
        labels_y = torch.cat(actuals_y)
        labels_d = torch.cat(actuals_d)

        if not empty_zx:
            zy_adata, zd_adata, zx_adata = [anndata.AnnData(_) for _ in [zy, zd, zx]]
            adatas = [zy_adata, zd_adata, zx_adata]
        else:
            zy_adata, zd_adata = [anndata.AnnData(_) for _ in [zy, zd]]
            adatas = [zy_adata, zd_adata]

        name = ['zy', 'zd', 'zx']
        for i, _ in enumerate(adatas):
            _.obs['batch'] = patients[labels_d]
            _.obs['cell_type'] = cell_types[labels_y]
            # save_name_pat = '_diva_new_semi_sup_train_' + name[i] + '_by_batches_heldout_pat_' + str(test_patient) + '.png'
            # save_name_cell_type = '_diva_new_semi_sup_train_' + name[i] + '_by_label_heldout_pat_' + str(test_patient) + '.png'
            save_name = f"_{model_name}_train+test_set_{name[i]}.png"

            sc.pp.neighbors(_, use_rep="X", n_neighbors=15)
            sc.tl.umap(_, min_dist=.3)
            sc.pl.umap(_, color=['batch', 'cell_type'], size=15, alpha=.8, save=save_name)

if __name__ == "__main__":
    model_name = get_valid_diva_models()[0]

    model = torch.load(model_name + ".model")
    args = torch.load(model_name + ".config")

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

    train_loader, test_loader, crc_adata = load_rcc_to_crc_data_loaders(shuffle=False)

    cell_types = test_loader.cell_types
    patients = test_loader.patients

    train_loader = data_utils.DataLoader(
        train_loader,
        batch_size=args.batch_size,
        shuffle=False)
    test_loader = data_utils.DataLoader(
        test_loader,
        batch_size=args.batch_size,
        shuffle=False)

    crc_adata.obs['cell_types'] = np.array(load_louvain().cell_types)
    crc_adata.obsm['X_umap'] = np.array(load_umap())

    plot_umap(train_loader, test_loader, model, cell_types, patients, model_name, device)
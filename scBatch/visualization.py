"""
This is  script for all our visualization needs

"""
import torch
import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import anndata

from .helper_functions import ensure_dir

def save_cm(true, preds, name, reduce_cm=True, label_names=None, sort_labels=False):
    """

    Parameters
    ----------
    true : list list (np.array)
        true labels for a cm
    preds : list like (np.array)
        predictions for a cm
    name : str
        name of figure (without .png ext)
    reduce_cm : bool
        toggle to get rid of 0/NA cols and rows
    label_names : list
        if the true and preds are ints, provide a list of names that map the ints to labels
    sort_labels : bool
        if the true and preds args are lists of names, choose to sort them by alphabetical order

    Returns
    -------

    """
    if label_names is not None:
        cm = confusion_matrix(true, preds, normalize='true')
        labels = label_names
    else:
        if sort_labels:
            labels = np.unique(np.concatenate([true, preds]))
        else:
            labels = pd.factorize(np.concatenate([true, preds]))[1]
        cm = confusion_matrix(true, preds, normalize='true', labels=labels)
    cm_norm_df = pd.DataFrame(cm, index=labels, columns=labels)
    if reduce_cm:
        cm_norm_df = cm_norm_df.dropna(axis=0, how='all')
        cm_norm_df = cm_norm_df[~(cm_norm_df == 0).all(axis=1)]
        cm_norm_df = cm_norm_df.T[~(cm_norm_df == 0).all(axis=0)].T
    plt.figure(figsize=(cm_norm_df.shape[1], cm_norm_df.shape[0]))
    ax = sns.heatmap(cm_norm_df, cmap="YlGnBu", vmin=0, vmax=1,
                     linewidths=.5, annot=True, fmt='4.2f', square=True)
    ax.figure.tight_layout()
    ensure_dir("./cm_figs")
    save_name = f"./cm_figs/cm_{name}.png"
    plt.savefig(save_name)


def plot_embeddings(model, data_loaders, device, fig_name):
    empty_zx = False
    patients = data_loaders['sup'].dataset.domains
    cell_types = data_loaders['sup'].dataset.labels

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
            if not empty_zx:
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
        if not empty_zx:
            zx = np.vstack(zx_)
        labels_y = np.hstack(actuals_y)
        labels_d = np.hstack(actuals_d)
        if not empty_zx:
            zy_adata, zd_adata, zx_adata = [anndata.AnnData(_) for _ in [zy, zd, zx]]
            adatas = [zy_adata, zd_adata, zx_adata]
        else:
            zy_adata, zd_adata = [anndata.AnnData(_) for _ in [zy, zd]]
            adatas = [zy_adata, zd_adata]
        name = ['zy', 'zd', 'zx']
        train_labels = patients[labels_d]
        zy_adata.obs['batch'] = train_labels
        zy_adata.obs['cell_type'] = cell_types[labels_y]
        zd_adata.obs['batch'] = train_labels
        zd_adata.obs['cell_type'] = cell_types[labels_y]
        train_cell_type_encoding = zy_adata
        train_batch_encoding = zd_adata
        for i, _ in enumerate(adatas):
            _.obs['batch'] = patients[labels_d]
            _.obs['cell_type'] = cell_types[labels_y]
            save_name = f"_{fig_name}_train_set_{name[i]}.png"
            sc.pp.neighbors(_, use_rep="X", n_neighbors=15)
            sc.tl.umap(_, min_dist=.3)
            sc.pl.umap(_, color=['batch', 'cell_type'], save=save_name)
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
            if not empty_zx:
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
        if not empty_zx:
            zx = np.vstack(zx_)
        labels_y = np.hstack(actuals_y)
        labels_d = np.hstack(actuals_d)
        if not empty_zx:
            zy_adata, zd_adata, zx_adata = [anndata.AnnData(_) for _ in [zy, zd, zx]]
            adatas = [zy_adata, zd_adata, zx_adata]
        else:
            zy_adata, zd_adata = [anndata.AnnData(_) for _ in [zy, zd]]
            adatas = [zy_adata, zd_adata]
        name = ['zy', 'zd', 'zx']
        test_labels = patients[labels_d]
        zy_adata.obs['batch'] = test_labels
        zy_adata.obs['cell_type'] = cell_types[labels_y]
        zd_adata.obs['batch'] = test_labels
        zd_adata.obs['cell_type'] = cell_types[labels_y]
        test_cell_type_encoding = zy_adata
        test_batch_encoding = zd_adata
        for i, _ in enumerate(adatas):
            _.obs['batch'] = patients[labels_d]
            _.obs['cell_type'] = cell_types[labels_y]
            save_name = f"_{fig_name}_test_set_{name[i]}.png"
            sc.pp.neighbors(_, use_rep="X", n_neighbors=15)
            sc.tl.umap(_, min_dist=.3)
            sc.pl.umap(_, color=['batch', 'cell_type'], save=save_name)
    full_zy = train_cell_type_encoding.concatenate(test_cell_type_encoding)
    full_zd = train_batch_encoding.concatenate(test_batch_encoding)
    all_patients = np.hstack([train_labels, test_labels])
    full_zy.obs['batch'] = all_patients
    full_zd.obs['batch'] = all_patients
    sc.pp.neighbors(full_zy, n_neighbors=15)
    sc.pp.neighbors(full_zd, n_neighbors=15)
    sc.tl.umap(full_zy, min_dist=.3)
    sc.tl.umap(full_zd, min_dist=.3)
    sc.pl.umap(full_zy, color=['batch', 'cell_type'], save=f"_{fig_name}_train+test_zy.png")
    sc.pl.umap(full_zd, color=['batch', 'cell_type'], save=f"_{fig_name}_train+test_zd.png")
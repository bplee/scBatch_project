from scnym.api import scnym_api
import torch
import os
import numpy as np
import anndata
import sys
import scanpy as sc
import pandas as pd
import scnym

import urllib
import json

# allow tensorboard outputs even though TF2 is installed
# broke the tensorboard/pytorch API
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

from ForBrennan.DIVA.dataset.rcc_loader_semi_sup import RccDatasetSemi


def get_balanced_classes(adata):
    """
    Takes the n = |smallest size class of adata|, and draws random groups of size n for each class
    and returns the random groups

    Parameters
    ----------
    adata: anndata.Anndata obj
        anndata obj with entries in adata.obs['cell_type']

    Returns
    -------
    anndata.Anndata obj
        Smaller anndata obj with balanced class distribution
    """
    """
    Returns an anndata obj with balanced labels

    Input:
        anndata obj with stuff in adata.obs['cell_type']

    Returns:
        Smaller anndata obj with balanced labels

    """
    counts = adata.obs.cell_type.value_counts()
    min_cell_type, min_num = counts.index[-1], counts[-1]
    rtn = []
    for i, cell_type_count in enumerate(counts):
        cell_type = counts.index[i]
        cell_type_inds = np.array([i for i, val in enumerate(adata.obs.cell_type == cell_type) if
                                   val])  # this line is returning the inds of all points with given cell type
        a = np.random.choice(cell_type_count, min_num, replace=False)  # choose n points from the list
        n_inds = cell_type_inds[a]  # sampled n random indices
        rtn.extend(n_inds)
    return adata[rtn, :]

if __name__ == "__main__":
    from .quick_prescnym_data_load import get_Rcc_adata

    # getting training and testing data
    TEST_PATIENT = 2
    TRAIN_PATIENT = 4
    X_DIM = 10000# 784 is the magic number for DIVA

    adata, data_obj = get_Rcc_adata(
        test_patient=TEST_PATIENT,
        x_dim=X_DIM
    )

    # blurb = """
    # Loaded 3 useful annData objects:
    #     - adata (ready as input to scnym)
    #     - train_adata
    #     - test_adata
    # Ready to train scnym
    # """
    # print(blurb)

    # now we want to create a balanced train/testing set to check the data

    patients = data_obj.patients

    by_pat = [adata[adata.obs.patient==p] for p in patients]
    all_types_by_pat = pd.DataFrame([p.obs.cell_type.value_counts() for p in by_pat]).T

    all_types_by_pat > 1000

    easy_cell_list = ['CD8 Exhausted_A', 'Naive CD8/CD4', 'Treg']

    test_adata = adata[adata.obs.patient==patients[TEST_PATIENT]]
    train_adata = adata[adata.obs.patient==patients[TRAIN_PATIENT]]

    train_adata = train_adata[train_adata.obs.cell_type.isin(easy_cell_list)]
    test_adata = test_adata[test_adata.obs.cell_type.isin(easy_cell_list)]

    test_adata.obs.annotations = 'Unlabeled'

    exp_adata = train_adata.concatenate(test_adata)

    scnym_api(adata=exp_adata,
              task='train',
              config='no_new_identity',
              out_path='./scnym_mini_exp_output',  # this is going in WORKING DIR
              groupby='annotations')
    print("Done training. Now for prediction")
    scnym_api(
        adata=exp_adata,
        task='predict',
        key_added='scNym',
        config='no_new_identity',
        trained_model='./scnym_mini_exp_output'
    )

    sc.pp.neighbors(exp_adata, use_rep='X_scnym', n_neighbors=30)
    sc.tl.umap(exp_adata, min_dist=.3)
    # the following are the scnym internal embeddings colored by batch and cell type
    sc.pl.umap(exp_adata, color='patient', size=5, alpha=.2, save='_scnym_mini_exp_embedding_patient.png')
    sc.pl.umap(exp_adata, color='cell_type', size=5, alpha=.2, save='_scnym_mini_exp_embedding_celltype.png')

    sc.pp.neighbors(exp_adata, use_rep='X', n_neighbors=30)
    sc.tl.umap(exp_adata, min_dist=.3)
    sc.pl.umap(exp_adata, color='cell_type', size=5, alpha=.2, save='_scnym_mini_exp_og_data_umap_celltype.png')
    sc.pl.umap(exp_adata, color='patient', size=5, alpha=.2, save='_scnym_mini_exp_og_data_umap_patient.png')

    print("While this isn't a run to test balanced classes, its a run of a more balanced easier experiment that scnym should perform well on")

    # balanced_test_adata = get_balanced_classes(test_adata)
    # print("created balanced_adata from test_adata")
    #
    # get_balanced_classes(train_adata)



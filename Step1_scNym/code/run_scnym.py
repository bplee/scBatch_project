from scnym.api import scnym_api
import torch
import os
import numpy as np
import anndata
import sys
from scvi.dataset import GeneExpressionDataset
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

# changing directory to project dir
print(os.getcwd())
WORKING_DIR = "/data/leslie/bplee/scBatch"

# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

# from ForBrennan.DIVA.dataset.rcc_loader_semi_sup import RccDatasetSemi
from Step0_Data.code.pkl_load_data import PdRccAllData

from new_prescnym_data_load import get_Rcc_adata

# to balance the test distribution
def get_balanced_classes(adata):
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
        cell_type_inds = np.array([i for i,val in enumerate(adata.obs.cell_type == cell_type) if val]) # this line is returning the inds of all points with given cell type
        a = np.random.choice(cell_type_count, min_num, replace=False) # choose n points from the list
        n_inds = cell_type_inds[a] # sampled n random indices
        rtn.extend(n_inds)
    return adata[rtn,:]


def train_scnym_model(adata, outpath,
                config='no_new_identity',
                groupby='annotations'):
    """
    Runs scnym training procedure

    Parameters
    ----------
    adata : AnnData obj
        data object holding LOG NORMALIZED counts, labels, and domain (patient, binary labels)
    outpath : str
        file path to directory to save output to (doesn't have to exist) (e.g. `./201025_scnym_test_output')
    config : str, optional
        allows user to change different modes
        (default is 'no_new_identity', ie. not expecting any unseen cell types)
    groupby : str, optional
        column in adata.obs where training labels are specified, and test labels are set to 'Unlabeled'
        (default is 'annotations')

    Returns
    -------
    None, trains an scnym model and saves output to outpath
    """

    scnym_api(adata=adata,
              task='train',
              config=config,
              out_path=outpath,
              groupby=groupby)

def predict_from_scnym_model(adata, trained_model,
                     key_added='scNym',
                     config='no_new_identity'):
    """
    Makes cell type predictions for a matrix of counts from previously trained scnym model

    Parameters
    ----------
    adata : AnnData obj
        matrix of counts stored in adata.X with
    trained_model : str
        filepath to directory with a previously trained model
    key_added : str, optional
        name of column to be added to adata with predictions
        (default is 'scNym')
    config : str, optional
        allows user to change different modes
        (default is 'no_new_identity', ie. not expecting any unseen cell types)

    Returns
    -------
    None,
        adds new column to adata object with column name `key_added`

    """

    scnym_api(
        adata=adata,
        task='predict',
        key_added=key_added,
        config=config,
        trained_model=trained_model
    )

def plot_scnym_umap(adata, use_rep='X_scnym', color_labeling='scNym'):
    """
    Plots umap embedding, colored by choice of labling

    Parameters
    ----------
    adata : AnnData obj
        post prediction, ie. `X_scnym`
    use_rep : str, optional
        raw vector data that you want to create a umap embedding for (needs to be in
    color_labeling : str, optional

    Returns
    -------

    """
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=30)
    sc.tl.umap(adata, min_dist=.3)
    # the following are the scnym internal embeddings colored by batch and cell type
    sc.pl.umap(adata, color=color_labeling, size=5, alpha=.2, save='scnym_embedding_batch.png')

if __name__ == "__main__":
    print(f"Current Working Dir: {os.getcwd()}")
    outpath = "./201025_scnym_temp_output"

    adata, data_obj = get_Rcc_adata(test_patient=5, train_patient=4, x_dim=784)
    train_scnym_model(adata, outpath)
    predict_from_scnym_model(adata, outpath)
    

# old balancing classes for scnym code
# TODO: shouldnt do this re naming, should just change the test/train patients
# balanced_train_adata = get_balanced_classes(test_adata)
# print("created balanced_train_adata from test_adata")
#
# balanced_test_adata = get_balanced_classes(train_adata)
# print("created balanced_test_adata from train_adata")


#fixing annotations because we switched the train and test sets
# balanced_train_adata.obs.annotations = balanced_train_adata.obs.cell_type
# balanced_test_adata.obs.annotations = "Unlabeled"
#
# adata = balanced_train_adata.concatenate(balanced_test_adata)

# old hand coded training code
# training scnym
# scnym_api(adata=adata,
#           task='train',
#           config='no_new_identity',
#           out_path='./scnym_test_output',  # this is going in WORKING DIR
#           groupby='annotations')
# print("Done training. Now for prediction")
# scnym_api(
#     adata=adata,
#     task='predict',
#     key_added='scNym',
#     config='no_new_identity',
#     trained_model='./scnym_test_output'
# )

# sc.pp.neighbors(adata, use_rep='X_scnym', n_neighbors=30)
# sc.tl.umap(adata, min_dist=.3)
# # the following are the scnym internal embeddings colored by batch and cell type
# sc.pl.umap(adata, color='batch', size=5, alpha=.2, save='scnym_embedding_batch.png')
# sc.pl.umap(adata, color='cell_type', size=5, alpha=.2, save='scnym_embedding_celltypes.png')
# # sc.pl.umap(adata, color='X_scnym', size=5, alpha=.2, save='201022_scnym_embedding_celltypes.png')
#
# sc.pp.neighbors(adata, use_rep='X', n_neighbors=30)
# sc.tl.umap(adata, min_dist=.3)
# sc.pl.umap(adata, color='cell_type', size=5, alpha=.2, save='scnym_og_data_umap_celltype.png')

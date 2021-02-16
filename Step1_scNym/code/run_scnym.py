from scnym.api import scnym_api
import torch
import os
import numpy as np
import argparse
import anndata
import sys
from scvi.dataset import GeneExpressionDataset
import scanpy as sc
import pandas as pd
import scnym
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
from Step0_Data.code.starter import *

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

def get_accuracies(adata, key_added="scNym", test_patient=None):
    """
    Used to get the accuracy and weighted accuracy of scnym predictions
    If test_patient arg is submitted, makes a cm matrix in `cm_figs/` dir

    Parameters
    ----------
    adata : annData object
        assumes already run a prediction

    key_added : str
        default is "scNym"
        this is the name of the column in adata.obs with the annotations

    Returns
    -------
    tuple: accuracy and weighted accuracy of the predictions
    saves cm matrix

    """
    cell_types = np.unique(adata.obs.cell_type)
    patients = np.unique(adata.obs.batch)
    test_indices = adata.obs.annotations == "Unlabeled"
    preds = adata.obs[key_added][test_indices]
    golden_labels = adata.obs.cell_type[test_indices]

    preds_ints = np.empty(len(preds))
    golden_labels_ints = np.empty(len(golden_labels))

    for i, c in enumerate(cell_types):
        idx_preds = np.where(preds == c)[0]
        preds_ints[idx_preds] = i
        idx_labels = np.where(golden_labels == c)
        golden_labels_ints[idx_labels] = i

    accuracy = sum(preds_ints == golden_labels_ints)/len(preds)
    cm = confusion_matrix(golden_labels_ints, preds_ints)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    weighted_accuracy = np.mean(np.diag(cm_norm))

    if test_patient is not None:
        ensure_dir("cm_figs")
        print("Making confusion matrix")
        cm_norm_df = pd.DataFrame(cm_norm, index=cell_types, columns=cell_types)
        plt.figure(figsize=(20, 20))
        ax = sns.heatmap(cm_norm_df, cmap="YlGnBu", vmin=0, vmax=1, linewidths=.5, annot=True, fmt='4.2f', square=True)
        plt.savefig('cm_figs/fig_scnym_cm_test_pat_' + str(test_patient) + '.png')

    return accuracy, weighted_accuracy



def plot_scnym_umap(adata, test_pat, train_pat=None, use_rep='X_scnym'):
    """
    Plots umap embedding, colored by choice of labling

    Parameters
    ----------
    adata : AnnData obj
        needs to be post prediction, ie. `X_scnym` is stored in adata
    use_rep : str, optional
        raw vector data that you want to create a umap embedding for (needs to be in adata)
        default is X_scnym (the embedding representation of the log normalized counts
    color_labeling : str, optional
        color that needs t
        default is 'scNym' as `predict_from_scnym_model` has 'scNym' as default for `key_added` param
    save_name : str, optional
        default is 'scnym_embedding.png'
        should change name to represent the labeling color
        this saves automatically to ./figures/umapscnym_embedding.png [sic]

    Returns
    -------
    None
        saves two figures

    """
    if train_pat is None:
        train_pat = "ALL"
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=30)
    sc.tl.umap(adata, min_dist=.3)
    save_name = f"_scnym_train_domain_{test_pat}_test_domain_{train_pat}_batches+celltype.png"
    sc.pl.umap(adata, color=['batch', 'cell_type'], size=5, alpha=.2, save=save_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scANVI')
    parser.add_argument('--test_patient', type=int, default=5,
                        help='test domain')
    parser.add_argument('--train_patient', type=int, default=None,
                        help='test domain')
    args_scnym = parser.parse_args()
    print(args_scnym)

    print(f"Current Working Dir: {os.getcwd()}")

    train_pat = args_scnym.train_patient
    test_pat = args_scnym.test_patient

    outpath = f"201117_scnym_SSL_test_pat_{test_pat}"

    adata, data_obj = get_Rcc_adata(test_patient=test_pat, train_patient=train_pat, x_dim=784)
    print(f"Training scNym model off training patient {train_pat}, with test patient {test_pat}")
    train_scnym_model(adata, outpath)
    print(f"Saved model to {outpath}")
    print(f"Predicting training and testing set")
    predict_from_scnym_model(adata, trained_model=outpath)
    accur, weighted_accur = get_accuracies(adata)
    print(f"Accuracy: {accur}\nWeigted Accuracy: {weighted_accur}")
    plot_scnym_umap(adata, test_pat)

    # accurs, weighted_accurs = [],[]
    # for test_pat in range(6):
    #     model_name = f"210202_multi_domain_test_pat_{test_pat}"
    #     adata, obj = get_Rcc_adata(test_pat, x_dim=784)
    #     predict_from_scnym_model(adata, model_name)
    #     out = get_accuracies(adata, test_patient=test_pat)
    #     accurs.append(out[0])
    #     weighted_accurs.append(out[1])
    # print(accurs, weighted_accurs)
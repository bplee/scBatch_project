import anndata
import sys
import os
import scanpy as sc
import numpy as np
import pandas as pd
from scvi.dataset import GeneExpressionDataset
import scipy

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

from Step0_Data.code.starter import *
from Step6_RCC_to_CRC.code.rcc_to_crc_test import get_diva_loaders

# this data is from: https://zenodo.org/record/4263972#.YFjtJS1h1B0


TIC_DATA_PATH = "/data/leslie/bplee/scBatch/TIC_atlas/data/TICAtlas.h5ad"
TIC_META_DATA_PATH = "/data/leslie/bplee/scBatch/TIC_atlas/data/TICAtlas_metadata.csv"
TIC_DOWNSAMPLED_PATH = "/data/leslie/bplee/scBatch/TIC_atlas/data/TICAtlas_downsampled_1000.h5ad"

def load_data(data_path=TIC_DATA_PATH):
    """
    loads TIC atlas data [317111 x 87659]

    Parameters
    ----------
    data_path

    Returns
    -------
    anndata.Dataframe
        has sparse matrix in the count part
        has a 'percent.mt' column
        gender, source (where the sample comes from), subtype (tumor type)
    """
    return anndata.read_h5ad(data_path)

def load_meta_data(meta_data_path=TIC_META_DATA_PATH):
    return pd.read_csv(meta_data_path, index_col=0)


def load_downsampled(data_path=TIC_DOWNSAMPLED_PATH):
    return anndata.read_h5ad(data_path)


def clean_tic(adata):
    """

    Parameters
    ----------
    adata : anndata.AnnData
        input from load adata

    Returns
    -------
    anndata.AnnData

    """
    print(f"Adata Starting Shape: {adata.shape}")
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    adata.var['ribo'] = adata.var_names.str.startswith(("RP"))
    # no mt genes in this data, but they have percent mt from prior analysis
    print(f" Number of MT genes: {sum(adata.var['mt'])} / {adata.shape[1]}")
    print(f" Number of Ribo genes: {sum(adata.var['ribo'])} / {adata.shape[1]}")
    print(f" Removing ribo and mitochondrial genes")

    print(f" Number of cells with MT%>20: {sum(adata.obs['percent.mt'] > 20)} ")
    adata = adata[adata.obs['percent.mt'] < 20, :]

    extra_removing = ["NEAT1", "MALAT1"]
    print(f" Removing {extra_removing} from list")
    extra_bools = adata.var_names.isin(extra_removing)  # these are true for NEAT1 and MALAT1
    keep_genes = ~adata.var.mt & ~adata.var.ribo & ~extra_bools

    adata = adata[:, keep_genes]
    print(f"Adata Ending Shape: {adata.shape}")

    return adata

def get_label_counts(adata, domain_name="subtype", label_name="cell_type"):
    return adata.obs[[domain_name, label_name]].value_counts(sort=False).to_frame().pivot_table(index=domain_name,
                                                                                               columns=label_name,
                                                                                               fill_value=0).T


def set_adata_train_test_batches(adata, test, train=None, label_name="subtype"):
    """
    Gives back adata with training ("0") and test ("1") labels specified in adata.obs.batch

    Parameters
    ----------
    adata : anndata.AnnData
    test : list
        contains integers corresponding to which labels are going to be test domains
    train : list (default: None)
        contains integers corresponding to which labels are going to be train_domains
    label_name: str (default: "subtype")
        name of adata.obs column that contains information that you want to use to stratify domains

    Returns
    -------
    anndata.AnnData
        with added adata.obs.batch column with "0" for training data and "1" for test data

    """
    # creating the column
    adata.obs['batch'] = "0"

    # getting the ints:
    labels, label_map = pd.factorize(adata.obs[label_name])

    # make sure the type of test and train are lists:
    test = wrap(test)
    # mark all test data
    test_inds = np.isin(labels, test)
    adata.obs.batch[test_inds] = "1"

    if train is None:
        return adata
    else:
        train = wrap(train)
        train_inds = np.isin(labels, train)
        adata = adata[(train_inds | test_inds),:]
        return adata

if __name__ == "__main__":
    test_tumor_type = "BC"

    adata = load_data()
    adata = clean_tic(adata)
    gene_ds = GeneExpressionDataset()
    tumor_types = adata.obs.subtype
    gene_ds.populate_from_data(X=adata.X,
                               gene_names=np.array(adata.var.index),
                               batch_indices=pd.factorize(tumor_types)[0],
                               remap_attributes=False)
    gene_ds.subsample_genes(784)

    adata = adata[:, gene_ds.gene_names]
    # batches are going to be built off of adata.obs.subtype
    adata.obs['batch'] = "0"
    adata.obs.batch[tumor_types == test_tumor_type] = "1"
    adata.X = adata.X.toarray()
    a = get_diva_loaders(adata, domain_name="subtype", label_name="cell_type")

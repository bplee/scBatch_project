import pandas as pd
import numpy as np
import os
import anndata
import sys
import scanpy as sc
from scvi.dataset import GeneExpressionDataset

WORKING_DIR = "/data/leslie/bplee/scBatch_project"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")
from Step0_Data.code.starter import *
from scBatch.dataprep import set_adata_train_test_batches
from scBatch.main import *

data_dir = "../data/"
TIM_DATA_FILEPATH = "/data/leslie/bplee/scBatch_project/pan_cancer_TIMs/quickload_data/TIMs_all_data.h5ad"


def read_dataset(name):
    data = pd.read_csv(name+"_normalized_expression.csv")
    counts = data.drop(['index'], axis=1)
    metadata = pd.read_csv(name+"_metadata.csv")
    adata = anndata.AnnData(X=counts, obs=metadata)
    return adata


def get_valid_datasets(dir_path):
    files = os.listdir(dir_path)
    lst = []
    rtn = []
    for f in files:
        if suffix(f, suf='_metadata.csv'):
            lst.append(suffix(f, suf='_metadata.csv'))
        if suffix(f, suf="_normalized_expression.csv"):
            lst.append(suffix(f, suf="_normalized_expression.csv"))
    while lst:
        curr = lst.pop()
        if os.path.exists(dir_path+curr+"_metadata.csv") and os.path.exists(dir_path+curr+"_metadata.csv"):
            rtn.append(curr)
            lst.remove(curr)
    return rtn


def suffix(word, suf):
    if word[-len(suf):] == suf:
        return word[:-len(suf)]
    return False


def load_data(data_dir):
    rtn = []
    datasets = get_valid_datasets(data_dir)
    print(f" found {len(datasets)} datasets in {data_dir}")
    for name in get_valid_datasets(data_dir):
        rtn.append(read_dataset(os.path.join(data_dir, name)))
        print(f" added {name} dataset")
    return rtn


def quick_load(filepath=TIM_DATA_FILEPATH):
    return anndata.read_h5ad(filepath)


def identify_singleton_labels(label_count_table):
    # temp = get_label_counts(adata.obs, label_name, domain_name)
    temp = label_count_table > 0
    # these are cell types that are only present in one patient
    cell_type_cancer_prevalence = temp.sum(axis=1)[0]
    return list(cell_type_cancer_prevalence[cell_type_cancer_prevalence == 1].index)


def remove_prefixes(cell_names):
    return cell_names.map(lambda x: x[4:])


def filter_cancers(adata, cancers_types_to_remove=["L", "OV", "PACA", "MM", "LYM"]):
    bool_inds = ~adata.obs.cancer.isin(cancers_types_to_remove)
    print(f'removing {sum(~bool_inds)} cells')
    return adata[bool_inds, :]


def filter_cell_types(adata, cell_types_to_remove):
    bool_inds = ~adata.obs.MajorCluster.isin(cell_types_to_remove)
    print(f'removing {sum(~bool_inds)} cells')
    return adata[bool_inds, :]


if __name__ == "__main__":
    adata = quick_load(TIM_DATA_FILEPATH)
    print(f" loaded all TIM data into anndata obj named: `adata`")
    print(" Removing prefixes")
    adata.obs.MajorCluster = remove_prefixes(adata.obs.MajorCluster)

    adata = filter_cancers(adata)
    label_counts = get_label_counts(adata.obs, "MajorCluster", "cancer")
    cell_types_to_remove = identify_singleton_labels(label_counts)
    adata = filter_cell_types(adata, cell_types_to_remove)

    sc.normalize_total(adata, 1e5)

    gene_ds = GeneExpressionDataset()
    batches = adata.obs.patient
    gene_ds.populate_from_data(X=adata.X,
                               gene_names=np.array(adata.var.index),
                               batch_indices=pd.factorize(batches)[0],
                               remap_attributes=False)
    gene_ds.subsample_genes(784)

    adata = adata[:, gene_ds.gene_names]
    # batches are going to be built off of adata.obs.subtype
    adata = set_adata_train_test_batches(adata,
                                         test=0,
                                         train=None,
                                         domain_name="cancer")

    adata.obs['cell_type'] = adata.obs['MajorCluster'].copy()
    del adata.obs['MajorCluster']

    adata.obs['domain'] = adata.obs['cancer'].copy()

    # obj = DIVAObject()
    # obj.args.epochs=10
    # obj.fit(adata, '210624_test')
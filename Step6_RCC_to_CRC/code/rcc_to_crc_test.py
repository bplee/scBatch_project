import sys
WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

import pandas as pd
import numpy as np
from Step0_Data.code.pkl_load_data import PdRccAllData
import anndata
import torch
import torch.utils.data as data_utils
from scvi.dataset import GeneExpressionDataset
import scanpy as sc
from CRC_dataset.code.crc_data_load import *
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

def get_diva_loaders(adata, shuffle=False):
    """

    Parameters
    ----------
    adata : needs `patients` and `cell_types` columns

    Returns
    -------

    """
    if 'log1p' not in adata.uns:
        print("Looks you haven't taken the log of the data, doing it for you")
        sc.pp.log1p(adata)
    train_inds = adata.obs.batch == "0"
    test_inds = ~train_inds
    n_train = sum(train_inds)
    n_test = sum(test_inds)

    data = adata.X
    patients, patient_map = pd.factorize(adata.obs.patient)
    labels, label_map = pd.factorize(adata.obs.cell_type)

    data_train = data[train_inds,:]
    data_test = data[test_inds,:]
    labels_train = labels[train_inds]
    labels_test = labels[test_inds]
    batch_train = patients[train_inds]
    batch_test = patients[test_inds]

    if shuffle:
        # Shuffle everything one more time
        inds = np.arange(n_train)
        np.random.shuffle(inds)
        data_train = data_train[inds]
        labels_train = labels_train[inds]
        batch_train = batch_train[inds]
        inds = np.arange(n_test)
        np.random.shuffle(inds)
        data_test = data_test[inds]
        labels_test = labels_test[inds]
        batch_test = batch_test[inds]

    # converting to tensors
    data_train = torch.as_tensor(data_train)
    data_test = torch.as_tensor(data_test)
    labels_train = torch.as_tensor(labels_train.astype(int))
    labels_test = torch.as_tensor(labels_test.astype(int))
    batch_train = torch.as_tensor(batch_train.astype(int))
    batch_test = torch.as_tensor(batch_test.astype(int))

    # Convert to onehot
    n_labels = len(label_map)
    y = torch.eye(n_labels)
    labels_train = y[labels_train]
    labels_test = y[labels_test]

    # Convert to onehot
    n_pats = len(patient_map)
    d = torch.eye(n_pats)
    batch_train = d[batch_train]
    batch_test = d[batch_test]

    train_data_loader, test_data_loader = EmptyDIVALoader(), EmptyDIVALoader()

    train_data_loader.train = True
    test_data_loader.train = False

    train_data_loader.train_data, test_data_loader.test_data        = data_train.unsqueeze(1), data_test.unsqueeze(1)
    train_data_loader.train_labels, test_data_loader.test_labels    = labels_train, labels_test
    train_data_loader.train_domain, test_data_loader.test_domain    = batch_train, batch_test
    train_data_loader.cell_types, test_data_loader.cell_types       = label_map, label_map
    train_data_loader.patients, test_data_loader.patients           = patient_map, patient_map

    return train_data_loader, test_data_loader

class EmptyDIVALoader(data_utils.Dataset):
    """
    This is for DIVA
    Counts get log normalized
    """
    def __init__(self, train=True):
        self.train = train
    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)
    def __getitem__(self, index):
        if self.train:
            x = self.train_data[index]
            y = self.train_labels[index]
            d = self.train_domain[index]
        else:
            x = self.test_data[index]
            y = self.test_labels[index]
            d = self.test_domain[index]
        return x, y, d


def load_rcc_to_crc_data_loaders(cell_types_to_remove=["Plasma"],old_load=False, shuffle=False):
    """
    Function made from scratch code, returns DIVA RCC training and CRC test loaders for SSL training as well as
    crc adata so that we can plot predictions on the adata
    Returns
    -------
    train loader, test loader, crc adata

    """
    pkl_path = "/data/leslie/bplee/scBatch/CRC_dataset/pkl_files/201204_CRC_data.pkl"
    all_data = pd.read_pickle(pkl_path)
    patient_subset = ["TS-101T",
                      "TS-104T",
                      "TS-106T",
                      "TS-108T",
                      "TS-109T",
                      "TS-125T"]
    og_pat_inds = all_data['PATIENT'].isin(patient_subset)
    og_data = all_data[og_pat_inds]

    crc_adata = clean_data_qc(og_data, old_load=old_load)

    crc_genes = set(crc_adata.var.index.values)

    crc_patient = crc_adata.obs.batch

    crc_adata.obs['cell_types'] = load_louvain().cell_types
    crc_adata.obs['chirag'] = load_louvain().chirag

    cells_to_remove = crc_adata.obs['chirag'].isin(cell_types_to_remove)
    crc_adata = crc_adata[~cell_types_to_remove,:]
    # crc_adata.obsm['X_umap'] = np.array(load_umap())

    # this needs to get the annotaions from diva
    # preparing UMAP for new pat:
    # crc_adata = anndata.AnnData(np.log(crc_raw_counts + 1))
    # crc_adata.obs['batch'] = np.array(crc_data_patient)
    # crc_adata.obs['annotations'] = 'Unlabeled'
    # sc.pp.neighbors(crc_adata, n_neighbors=10)
    # sc.tl.umap(crc_adata)
    # sc.pl.umap(crc_adata, color=[], save='markers')

    # RCC DATA
    # --------
    # loading training set RCC, removing ccRCC cells
    rcc_obj = PdRccAllData(labels_to_remove=["Ambiguous", "Megakaryocyte", "TAM/TCR (Ambiguos)", "CD45- ccRCC CA9+"])
    rcc_patient = rcc_obj.data.patient
    rcc_cell_type = rcc_obj.data.cell_type
    rcc_raw_counts = rcc_obj.data.drop(["cell_type", "patient"], axis=1)

    # these are the ensembl.gene names
    rcc_genes = set(rcc_raw_counts.columns.values)

    # comparing set of gene names:
    print(f" Unique CRC gene names: {len(crc_genes)}\n Unique RCC gene names: {len(rcc_genes)}")

    universe = crc_genes.intersection(rcc_genes)
    print(f" Genes in both datasets: {len(universe)}")

    crc_adata = crc_adata[:, np.array(list(universe))]

    # getting rid of non shared genes and making adata's
    crc_adata.obs['annotations'] = 'Unlabeled'

    rcc_raw_counts = rcc_raw_counts[universe]

    rcc_adata = anndata.AnnData(rcc_raw_counts)
    rcc_adata.obs['cell_type'] = rcc_cell_type
    rcc_adata.obs['annotations'] = rcc_cell_type
    del rcc_raw_counts

    adata = rcc_adata.concatenate(crc_adata)
    # adata.obs['batch'] = np.array(pd.concat([rcc_patient, crc_patient]))

    pats = np.append(np.array(rcc_patient), np.array(crc_patient))
    adata.obs['patient'] = pats
    gene_ds = GeneExpressionDataset()
    gene_ds.populate_from_data(X=adata.X,
                               gene_names=np.array(adata.var.index),
                               batch_indices=pd.factorize(pats)[0],
                               remap_attributes=False)
    gene_ds.subsample_genes(784)

    adata = adata[:, gene_ds.gene_names]

    train_loader, test_loader = get_diva_loaders(adata, shuffle=shuffle)

    return train_loader, test_loader, crc_adata

if __name__ == "__main__":

    train_loader, test_loader, crc_adata = load_rcc_to_crc_data_loaders()

    # svm code:
    # data = gene_ds.X
    # train_inds = adata.obs.batch=='0'
    # x = data[train_inds, :]
    # y, map = pd.factorize(adata.obs.cell_types[train_inds])
    #
    # test_x = data[~train_inds, :]

    # svm = LinearSVC()
    svm.fit(x, y)
    preds = map[svm.predict(test_x)]



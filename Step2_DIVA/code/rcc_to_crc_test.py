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
import mygene
import anndata
import torch
import torch.utils.data as data_utils
from scvi.dataset import GeneExpressionDataset
import scanpy as sc

class DIVALoader(data_utils.Dataset):
    """
    This is for DIVA
    Counts get log normalized
    """
    def __init__(self, adata, test_pat, x_dim=784):
        self.test_patient = test_pat
        self.x_dim = x_dim
        train_loader_data, test_loader_data = self.get_diva_train_test_loaders(adata, test_pat, x_dim)
        self.train_data, self.train_labels, self.train_domain, self.cell_types, self.patients = train_loader_data
        self.test_data, self.test_labels, self.test_domain, self.cell_types, self.patients = test_loader_data

    def get_diva_train_test_loaders(self, adata, test_pat, x_dim):
        """
        Parameters
        ----------
        adata : anndata.AnnData obj
            raw count matrix with patient and cell type labels

        test_pat : str
            string corresponding to the name of the patient you want to be the test patient

        Returns
        -------
        training and testing data loaders for DIVA
        """

        cell_types = adata.obs.cell_types
        patients = adata.obs.batch
        raw_counts = cell_types.X
        print('Re-writing labels and patients as indices')
        labels, self.cell_type_names = pd.factorize(cell_types)
        batch_indices, patient_names = pd.factorize(patients)
        gene_names = adata.var  # np array
        print('Importing gene expression ds')
        gene_dataset = GeneExpressionDataset()
        gene_dataset.populate_from_data(
            X=np.array(raw_counts),
            batch_indices=batch_indices,
            labels=labels,
            gene_names=gene_names,
            cell_types=cell_types,
            remap_attributes=False
        )
        del raw_counts
        gene_dataset.subsample_genes(x_dim)

        print('Making tensor batches')

        idx_batch_train = ~(gene_dataset.batch_indices == test_pat).ravel()
        idx_batch_test = (gene_dataset.batch_indices == test_pat).ravel()

        batch_train = gene_dataset.batch_indices[idx_batch_train].ravel()
        batch_test = gene_dataset.batch_indices[idx_batch_test].ravel()

        labels_train = gene_dataset.labels[idx_batch_train].ravel()
        labels_test = gene_dataset.labels[idx_batch_test].ravel()

        data_train = gene_dataset.X[idx_batch_train]
        data_test = gene_dataset.X[idx_batch_test]

        data_train = np.log(data_train + 1)
        data_test = np.log(data_test + 1)

        # what is the point of this line?
        data_train = data_train / np.max(data_train)
        data_test = data_test / np.max(data_test)

        n_train = len(labels_train)
        n_test = len(labels_test)
        data_train = torch.as_tensor(data_train)
        data_test = torch.as_tensor(data_test)

        labels_train = torch.as_tensor(labels_train.astype(int))
        labels_test = torch.as_tensor(labels_test.astype(int))

        batch_train = torch.as_tensor(batch_train.astype(int))
        batch_test = torch.as_tensor(batch_test.astype(int))

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

        # Convert to onehot
        y = torch.eye(n_labels)
        labels_train = y[labels_train]
        labels_test = y[labels_test]

        # Convert to onehot
        d = torch.eye(6)
        batch_train = d[batch_train]
        batch_test = d[batch_test]

        print("Summary: Training Data Loader")
        print(f" data_train.shape: {data_train.shape}")
        print(f" labels_train.shape: {labels_train.shape}")
        print(f" batch_train.shape: {batch_train.shape}")
        train_loader = [data_train.unsqueeze(1), labels_train, batch_train, cell_type_names, patient_names]
        print("Summary: Test Data Loader")
        print(f" test patient: {patient_names[test_pat]} (#{test_pat})")
        print(f" train patient: {test_pat}")
        print(f" data_test.shape: {data_test.shape}")
        print(f" labels_test.shape: {labels_test.shape}")
        print(f" batch_test.shape: {batch_test.shape}")
        test_loader = [data_test.unsqueeze(1), labels_test, batch_test, cell_type_names, patient_names]

        return train_loader, test_loader

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


if __name__ == "__main__":

    # CRC DATA
    # --------
    crc_dir = "/data/leslie/bplee/scBatchCRC_dataset/code"

    # loading testing set crc data
    crc_pkl_path = "/data/leslie/bplee/scBatch/CRC_dataset/pkl_files/201204_CRC_data.pkl"
    crc_all_data = pd.read_pickle(crc_pkl_path)

    # getting one test patien
    crc_test_pat = "TS-101T"
    crc_data = crc_all_data[crc_all_data['PATIENT'] == crc_test_pat]
    crc_cluster = crc_data.CLUSTER
    crc_patient = crc_data.PATIENT
    crc_raw_counts = crc_data.drop(["CLUSTER", "PATIENT"], axis=1)

    crc_gene_names = crc_raw_counts.columns.values

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
    rcc_obj = PdRccAllData(labels_to_remove=["Ambiguous", "Megakaryocyte", "CD45- ccRCC CA9+"])
    rcc_patient = rcc_obj.data.patient
    rcc_cell_type = rcc_obj.data.cell_type
    rcc_raw_counts = rcc_obj.data.drop(["cell_type", "patient"], axis=1)



    # these are the ensembl.gene names
    rcc_gene_names = rcc_raw_counts.columns.values

    # converting RCC ensembl.gene names to symbols
    mg = mygene.MyGeneInfo()
    rcc_gene_name_query = mg.querymany(rcc_gene_names, scope='ensembl.gene', fields='symbol', species='human')

    # getting list of symbol names
    rcc_gene_name_symbols = [a['symbol'] if 'symbol' in a else '' for a in rcc_gene_name_query]
    # rcc_gene_name_symbols = [a['symbol'] if 'symbol' in a else a['query'] for a in rcc_gene_name_query]

    # comparing set of gene names:
    crc_set = set(crc_gene_names)
    rcc_set = set(rcc_gene_name_symbols)
    print(f" Unique CRC gene names: {len(crc_set)}\n Unique RCC gene names: {len(rcc_set)}")

    genes_in_both = crc_set.intersection(rcc_set)
    print(f" Genes in both datasets: {len(genes_in_both)}")

    # making new pd df with new symbol names:
    # getting mapping from old names to new:
    mapping = {a['query']:a['symbol'] for a in rcc_gene_name_query if 'symbol' in a}
    rcc_symbol_genes = rcc_raw_counts.rename(columns=mapping)

    # getting rid of non shared genes and making adata's
    crc_raw_counts = crc_raw_counts[genes_in_both]
    crc_adata = anndata.AnnData(np.log(crc_raw_counts + 1))
    crc_adata.obs['annotations'] = 'Unlabeled'

    rcc_symbol_genes = rcc_symbol_genes[genes_in_both]
    # there are 2 cols for "BAZ2B" and "CYB561D2"
    df_cols = list(range(rcc_symbol_genes.shape[1]))
    inds_to_remove = []
    inds_to_remove.append(np.where(rcc_symbol_genes.columns == "BAZ2B")[0][1])
    inds_to_remove.append(np.where(rcc_symbol_genes.columns == "CYB561D2")[0][1])

    for i in inds_to_remove:
        df_cols.remove(i)

    rcc_symbol_genes = rcc_symbol_genes.iloc[:, df_cols]

    # df = pd.concat([rcc_symbol_genes, crc_raw_counts])
    # df_paients = pd.concat(rcc_patient, crc_patient)

    rcc_adata = anndata.AnnData(np.log(rcc_symbol_genes+1))
    rcc_adata.obs['cell_type'] = rcc_cell_type
    rcc_adata.obs['annotations'] = rcc_cell_type

    adata = rcc_adata.concatenate(crc_adata)
    adata.obs['batch'] = np.array(pd.concat([rcc_patient, crc_patient]))

    DIVALoader(adata, crc_test_pat)
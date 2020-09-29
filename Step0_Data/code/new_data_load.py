import os
import sys
import numpy as np
import pandas as pd
import pyreadr
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from scvi.dataset import GeneExpressionDataset
import rpy2.robjects as robjects
readRDS = robjects.r['readRDS']

WORKING_DIR = "/data/leslie/bplee/scBatch"
print("________CHANGING PATH_________")
sys.path.append(WORKING_DIR)
print("\tWorking dir appended to Sys path.")


class NewRccDatasetSemi(data_utils.Dataset):
    def __init__(self, test_patient, x_dim, train=True):
        self.test_patient = test_patient
        self.train = train
        self.x_dim = x_dim

        if self.train:
            self.train_data, self.train_labels, self.train_domain, self.cell_types, self.patients = self._get_data()
        else:
            self.test_data, self.test_labels, self.test_domain, self.cell_types, self.patients = self._get_data()

    def cell_types_batches(self):
        return self.cell_types, self.patients

    def _get_data(self):
        print('Getting data..')
        readRDS = robjects.r['readRDS']
        # pandas2ri.activate()

        annot = readRDS('/data/leslie/krc3004/RCC_Alireza_Sep2020/ccRCC_6pat_cell_annotations_June2020.rds')

        raw_counts = readRDS('/data/leslie/bplee/scBatch/Step0_data/data/200929_raw_counts.rds').transpose()

        cell_types = np.unique(annot.cluster_name)

        n_data_all = raw_counts.shape[0]
        n_gene_all = raw_counts.shape[1]

        print('re writing indices')

        labels = np.zeros([n_data_all, 1])
        for i, c in enumerate(cell_types):
            idx = np.where(annot.cluster_name.values == c)[0]
            labels[idx] = i
        labels = labels.astype(int)
        n_labels = len(np.unique(labels))

        patients = np.unique(annot.Sample)
        batch_indices = np.zeros([n_data_all, 1])
        for i, b in enumerate(patients):
            idx = np.where(annot.Sample.values == b)[0]
            batch_indices[idx] = i
        batch_indices = batch_indices.astype(int)

        gene_names = raw_counts.columns.values # np array

        n_each_cell_type = np.zeros(len(cell_types)).astype(int)
        for i in range(len(cell_types)):
            n_each_cell_type[i] = np.sum(labels == i)

        print('importing gene expression ds')

        gene_dataset = GeneExpressionDataset()
        gene_dataset.populate_from_data(
            X=raw_counts,
            batch_indices=batch_indices,
            labels=labels,
            gene_names=gene_names,
            cell_types=cell_types,
            remap_attributes=False
        )
        del raw_counts
        del annot
        gene_dataset.subsample_genes(self.x_dim)

        print('making tensor batches')


        idx_batch_train = ~(batch_indices == self.test_patient).ravel()
        idx_batch_test = (batch_indices == self.test_patient).ravel()

        batch_train = batch_indices[idx_batch_train].ravel()
        batch_test = batch_indices[idx_batch_test].ravel()

        labels_train = labels[idx_batch_train].ravel()
        labels_test = labels[idx_batch_test].ravel()

        data_train = gene_dataset.X[idx_batch_train]
        data_test = gene_dataset.X[idx_batch_test]

        data_train = np.log(data_train + 1)
        data_test = np.log(data_test + 1)

        data_train = data_train / np.max(data_train)
        data_test = data_test / np.max(data_test)

        data_train = np.reshape(data_train, [data_train.shape[0], int(np.sqrt(self.x_dim)), int(np.sqrt(self.x_dim))])
        data_test = np.reshape(data_test, [data_test.shape[0], int(np.sqrt(self.x_dim)), int(np.sqrt(self.x_dim))])

        n_train = len(labels_train)
        n_test = len(labels_test)

        print('run transformers')

        # Run transforms
        data_train = torch.as_tensor(data_train)
        data_test = torch.as_tensor(data_test)

        labels_train = torch.as_tensor(labels_train)
        labels_test = torch.as_tensor(labels_test)

        batch_train = torch.as_tensor(batch_train)
        batch_test = torch.as_tensor(batch_test)

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
        y = torch.eye(16)
        labels_train = y[labels_train]
        labels_test = y[labels_test]

        # Convert to onehot
        d = torch.eye(6)
        batch_train = d[batch_train]
        batch_test = d[batch_test]

        if self.train:
            return data_train.unsqueeze(1), labels_train, batch_train, cell_types, patients
        else:
            return data_test.unsqueeze(1), labels_test, batch_test, cell_types, patients

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
    
    # from ForBrennan.DIVA.dataset.rcc_loader_semi_sup import RccDatasetSemi

    # getting training and testing data
    TEST_PATIENT = 4
    X_DIM = 16323# 784 is the magic number for DIVA; 16323 is the max

    # getting training and testing data
    # data_obj = RccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=True, test=True, diva=False)
    print('getting it done')

    new_data_obj = NewRccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=True)

    # new_annot_file_path = '/data/leslie/krc3004/RCC_Alireza_Sep2020/ccRCC_6pat_cell_annotations_June2020.rds'
    # new_batch_corrected_data = '/data/leslie/krc3004/RCC_Alireza_Sep2020/ccRCC_6pat_June2020.rds'
    # raw_counts_file = '/data/leslie/bplee/scBatch/Step0_Data/data/200929_raw_counts.rds'
    #
    # annot = readRDS(new_annot_file_path)
    # raw_counts = readRDS(raw_counts_file).transpose() # this list of counts was generated by opening the seurat obj in R and saving the raw counts to a df for saveRDS


    # so this contains 31 labels instead of 16

    #data = readRDS(new_batch_corrected_data)

    

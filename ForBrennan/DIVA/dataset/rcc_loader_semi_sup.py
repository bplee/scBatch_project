"""Pytorch Dataset object that loads MNIST and SVHN. It returns x,y,s where s=0 when x,y is taken from MNIST."""

import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from scvi.dataset import GeneExpressionDataset
import pyreadr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri


class RccDatasetSemi(data_utils.Dataset):
    def __init__(self, test_patient, x_dim, train=True, diva=True, test=False):
        self.test_patient = test_patient
        self.train = train
        self.test = test
        self.x_dim = x_dim
        self.diva = diva #made this so sqrt(dim) doesnt have to be integer (line 120)

        self._get_data() # get all the data, training and/or test

        # if self.train and self.test:
        #     # normally this function returns stuff, but in this case, i just have the
        #     # functino interally set all the data since it was easier
        #     self._get_data()
        # else:
        #     if self.train:
        #         self.train_data, self.train_labels, self.train_domain, self.cell_types, self.patients = self._get_data()
        #     else:
        #         self.test_data, self.test_labels, self.test_domain, self.cell_types, self.patients = self._get_data()

    def cell_types_batches(self):
        return self.cell_types, self.patients

    def _get_data(self):
        
        annot_tam = pyreadr.read_r('/data/leslie/alireza/ccRCC/pat6_tams_annotations.rds')
        df_annot_tam = annot_tam[None]
        annot_tcell = pyreadr.read_r('/data/leslie/alireza/ccRCC/pat6_tcells_annotations.rds')
        df_annot_tcell = annot_tcell[None]
        df_annot_all_6_pat = df_annot_tcell.append(df_annot_tam, ignore_index=True)
        cell_type_tcell = np.unique(df_annot_tcell.cluster_name.values)
        cell_type_tam = np.unique(df_annot_tam.cluster_name.values)
        del annot_tam, df_annot_tam, annot_tcell, df_annot_tcell
        pandas2ri.activate()
        readRDS = robjects.r['readRDS']
        rawdata_tam = readRDS('/data/leslie/alireza/ccRCC/pat6_tams_rawcounts.rds').transpose()
        rawdata_tcell = readRDS('/data/leslie/alireza/ccRCC/pat6_tcells_rawcounts.rds').transpose()
        rawdata_all_6_pat = np.vstack((rawdata_tcell,rawdata_tam))
        del rawdata_tam, rawdata_tcell

        n_data_all = rawdata_all_6_pat.shape[0]
        n_gene_all = rawdata_all_6_pat.shape[1]

        cell_types = np.hstack((cell_type_tcell,cell_type_tam))
        labels = np.zeros([n_data_all,1])
        for i, c in enumerate(cell_types):
            idx = np.where(df_annot_all_6_pat.cluster_name.values == c)[0]
            labels[idx] = i
        labels = labels.astype(int)
        n_labels = len(np.unique(labels))

        patients = np.unique(df_annot_all_6_pat.Sample.values)
        batch_indices = np.zeros([n_data_all,1])
        for i, b in enumerate(patients):
            idx = np.where(df_annot_all_6_pat.Sample.values == b)[0]
            batch_indices[idx] = i
        batch_indices = batch_indices.astype(int)

        df = pd.read_csv("/data/leslie/alireza/ccRCC/gene_names.csv", header=0, index_col=0)
        gene_names = pd.Index(df.x.values)
        del df

        n_each_cell_type = np.zeros(len(cell_types)).astype(int)
        for i in range(len(cell_types)):
            n_each_cell_type[i] = np.sum(labels == i)

        gene_dataset = GeneExpressionDataset()
        gene_dataset.populate_from_data(
                X=rawdata_all_6_pat,
                batch_indices=batch_indices,
                labels=labels,
                gene_names=gene_names,
                cell_types=cell_types,
                remap_attributes = False
        )
        del rawdata_all_6_pat
        del df_annot_all_6_pat
        gene_dataset.subsample_genes(self.x_dim)

        # looks like the gene_dataset.X matrix gets downsampled but the bach indices and test indices don't
        # so those boolean numpy lists are of the og size (129097) but the gene_dataset.X arrays are of size (109221,50)


        # changing out the lines below so that it will make boolean numpy of correct size

        # idx_batch_train = ~(batch_indices == self.test_patient).ravel()
        # idx_batch_test = (batch_indices == self.test_patient).ravel()

        # batch_train = batch_indices[idx_batch_train].ravel()
        # batch_test = batch_indices[idx_batch_test].ravel()

        # labels_train = labels[idx_batch_train].ravel()
        # labels_test = labels[idx_batch_test].ravel()


        idx_batch_train = ~(gene_dataset.batch_indices == self.test_patient).ravel()
        idx_batch_test = (gene_dataset.batch_indices == self.test_patient).ravel()

        batch_train = gene_dataset.batch_indices[idx_batch_train].ravel()
        batch_test = gene_dataset.batch_indices[idx_batch_test].ravel()

        labels_train = gene_dataset.labels[idx_batch_train].ravel()
        labels_test = gene_dataset.labels[idx_batch_test].ravel()

        data_train = gene_dataset.X[idx_batch_train]
        data_test = gene_dataset.X[idx_batch_test]

        data_train = np.log(data_train + 1)
        data_test = np.log(data_test + 1)

        data_train = data_train / np.max(data_train)
        data_test = data_test / np.max(data_test)

        if self.diva:
            data_train = np.reshape(data_train, [data_train.shape[0], int(np.sqrt(self.x_dim)), int(np.sqrt(self.x_dim))])
            data_test = np.reshape(data_test, [data_test.shape[0], int(np.sqrt(self.x_dim)), int(np.sqrt(self.x_dim))])

        else: # dont make the output a list of square matrices, dont do anything
            pass

        n_train = len(labels_train)
        n_test = len(labels_test)

        # Run transforms
        data_train = torch.as_tensor(data_train)
        data_test = torch.as_tensor(data_test)

        # adding this line because of int type error when converting into tensor
        #labels_train = labels_train.astype("uint8")

        #labels_train = torch.as_tensor(labels_train)
        labels_train = torch.from_numpy(labels_train.astype("int"))
        #labels_test = torch.as_tensor(labels_test)
        labels_test = torch.from_numpy(labels_test.astype("uint8"))

        #batch_train = torch.as_tensor(batch_train)
        batch_train = torch.from_numpy(batch_train.astype("float"))
        #batch_test = torch.as_tensor(batch_test)
        batch_test = torch.from_numpy(batch_test.astype("float"))

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
        a = np.random.randint(0,15,(1000))
        labels_train = torch.nn.functional.one_hot(labels_train)
        labels_test = torch.nn.functional.one_hot(labels_test.long())

        # Convert to onehot
        d = torch.eye(6)
        batch_train = d[batch_train.long()]
        batch_test = d[batch_test.long()]

        self.cell_types = cell_types
        self.patients = patients

        if self.train:
            print('Returning Training data')
            print("data_train.shape:",data_train.shape)
            print("labels_train.shape:", labels_train.shape)
            print("batch_train.shape:", batch_train.shape)
            print("cell_types.shape:", cell_types.shape)
            print("patients.shape:", patients.shape)
            if self.diva:
                data_train = data_train.unsqueeze(1)
            self.train_data = data_train
            self.train_labels = labels_train
            self.train_domain = batch_train

        if self.test:
            print('Returning Testing data')
            print("data_test.shape:", data_test.shape)
            print("labels_test.shape:", labels_test.shape)
            print("batch_test.shape:", batch_test.shape)
            print("cell_types.shape:", cell_types.shape)
            print("patients.shape:", patients.shape)
            if self.diva:
                data_test.unsqueeze(1)
            self.test_data = data_test
            self.test_labels = labels_test
            self.test_domain = batch_test


        # else:
        #     if self.train:
        #         print("data_train.shape:",data_train.shape)
        #         print("labels_train.shape:", labels_train.shape)
        #         print("batch_train.shape:", batch_train.shape)
        #         print("cell_types.shape:", cell_types.shape)
        #         print("patients.shape:", patients.shape)
        #         if self.diva:
        #             data_train = data_train.unsqueeze(1)
        #         return data_train, labels_train, batch_train, cell_types, patients
        #     else:
        #         print("data_test.shape:", data_test.shape)
        #         print("labels_test.shape:", labels_test.shape)
        #         print("batch_test.shape:", batch_test.shape)
        #         print("cell_types.shape:", cell_types.shape)
        #         print("patients.shape:", patients.shape)
        #         if self.diva:
        #             data_test = data_test.unsqueeze(1)
        #         return data_test, labels_test, batch_test, cell_types, patients

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

    seed = 1

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    test_patient = 5

    train_loader = data_utils.DataLoader(
        RccDatasetSemi(test_patient, train=True),
        batch_size=100,
        shuffle=False)

    for i, (x, y, d) in enumerate(train_loader):

        if i == 0:
            print(y)
            print(d)


    test_loader = data_utils.DataLoader(
        RccDatasetSemi(test_patient, train=False),
        batch_size=100,
        shuffle=False)

    for i, (x, y, d) in enumerate(test_loader):

        if i == 0:
            print(y)
            print(d)

    

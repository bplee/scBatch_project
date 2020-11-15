print("Importing Modules")
import time
import os
import sys
import numpy as np
import pandas as pd
# import pyreadr
import torch
import torch.utils.data as data_utils
# from torchvision import datasets, transforms
from scvi.dataset import GeneExpressionDataset
# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri
# pandas2ri.activate()
# readRDS = robjects.r['readRDS']

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("CHANGING PATH:")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

from Step0_Data.code.pkl_load_data import PdRccAllData


class NewRccDatasetSemi(data_utils.Dataset):
    """
    This is for DIVA
    Counts get log normalized
    """
    def __init__(self, test_patient, x_dim, train=True, train_patient=None, starspace=False):
        self.test_patient = test_patient
        self.train = train
        self.x_dim = x_dim
        self.init_time = time.time()
        self.train_patient = train_patient
        self.starspace = starspace
        self.gene_names = None  # this is set in _get_data

        if self.starspace:
            # returns everything from one run, main difference is that it doesn't change the shape or convert to tensors
            self.train_data, self.train_labels, self.train_domain,\
            self.test_data, self.test_labels, self.test_domain,\
            self.cell_types, self.patients = self._get_data()
        else:
            if self.train:
                self.train_data, self.train_labels, self.train_domain, self.cell_types, self.patients = self._get_data()
            else:
                self.test_data, self.test_labels, self.test_domain, self.cell_types, self.patients = self._get_data()

    def cell_types_batches(self):
        return self.cell_types, self.patients

    @staticmethod
    def _convert_Rdf_to_pd(df):
        """
        Not working as intended because of pandas2riri2py() is buggy
        Meant to be a function that's used to ensure that you return a pd.DataFrame()
        Parameters
        ----------
        df: some data frame of data

        Returns
        -------
        pd.DataFrame obj of the data we want

        """
        if isinstance(df, pd.DataFrame):
            return df
        else:
            print(f"Oops, this df is of type {type(df)}")
            return pandas2ri.ri2py(df)

    def _get_data(self):
        print('Getting data..')
        # readRDS = robjects.r['readRDS']
        # pandas2ri.activate()

        print("Loading data from pkl...")
        # own data loader since R was being finicky
        data_obj = PdRccAllData()  # default args for this function will give me what I want
        raw_counts = data_obj.data.drop(['patient', 'cell_type'], axis=1)
        patients = data_obj.data.patient
        cell_types = data_obj.data.cell_type

        # print('Loading annotations...')
        # annot = readRDS('/data/leslie/krc3004/RCC_Alireza_Sep2020/ccRCC_6pat_cell_annotations_June2020.rds')
        # annot = self._convert_Rdf_to_pd(annot)
        # print("Loading raw counts...")
        # # raw_counts = readRDS('/data/leslie/bplee/scBatch/Step0_Data/data/200929_raw_counts.rds')
        # raw_counts = readRDS('/data/leslie/bplee/scBatch/Step0_Data/data/200929_raw_counts.rds').transpose()
        # raw_counts = self._convert_Rdf_to_pd(raw_counts)

        cell_type_names = np.unique(cell_types)

        n_data_all = raw_counts.shape[0]
        n_gene_all = raw_counts.shape[1]

        print('Re-writing labels and patients as indices')

        labels = np.zeros([n_data_all, 1])
        for i, c in enumerate(cell_type_names):
            idx = np.where(cell_types == c)[0]
            labels[idx] = i
        labels = labels.astype(int)
        n_labels = len(np.unique(labels))

        patient_names = np.unique(patients)
        batch_indices = np.zeros([n_data_all, 1])
        for i, b in enumerate(patient_names):
            idx = np.where(patients == b)[0]
            batch_indices[idx] = i
        batch_indices = batch_indices.astype(int)

        gene_names = raw_counts.columns.values # np array

        n_each_cell_type = np.zeros(len(cell_types)).astype(int)
        for i in range(len(cell_type_names)):
            n_each_cell_type[i] = np.sum(labels == i)

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
        del data_obj
        gene_dataset.subsample_genes(self.x_dim)
        #gene_dataset.filter_cells_by_count()

        self.gene_names = gene_dataset.gene_names

        print('Making tensor batches')

        if self.train_patient is None:
            idx_batch_train = ~(gene_dataset.batch_indices == self.test_patient).ravel()
        else:
            idx_batch_train = (gene_dataset.batch_indices == self.train_patient).ravel()
        idx_batch_test = (gene_dataset.batch_indices == self.test_patient).ravel()

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

        if self.starspace == True:
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

            return data_train, labels_train, batch_train, data_test, labels_test, batch_test, cell_type_names, patient_names



        # if we're running diva and not starspace we do other jazz
        else:

            data_train = np.reshape(data_train, [data_train.shape[0], int(np.sqrt(self.x_dim)), int(np.sqrt(self.x_dim))])
            data_test = np.reshape(data_test, [data_test.shape[0], int(np.sqrt(self.x_dim)), int(np.sqrt(self.x_dim))])

            print('Run transformers')

            # Run transforms
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

            self.return_time = time.time()
            print(f"Total Load Time: {self.return_time - self.init_time}")

            if self.train:
                print(f"train patient: {self.train_patient}")
                print(f"data_train.shape: {data_train.shape}")
                print(f"labels_train.shape: {labels_train.shape}")
                print(f"batch_train.shape: {batch_train.shape}")

                return data_train.unsqueeze(1), labels_train, batch_train, cell_type_names, patient_names
            else:
                print(f"test patient: {self.test_patient}")
                print(f"data_test.shape: {data_test.shape}")
                print(f"labels_test.shape: {labels_test.shape}")
                print(f"batch_test.shape: {batch_test.shape}")
                return data_test.unsqueeze(1), labels_test, batch_test, cell_type_names, patient_names

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
    X_DIM = 784

    # getting training and testing data
    # data_obj = RccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=True, test=True, diva=False)

    new_data_obj = NewRccDatasetSemi(test_patient=TEST_PATIENT, x_dim=X_DIM, train=True)

    annot_filepath = '/data/leslie/krc3004/RCC_Alireza_Sep2020/ccRCC_6pat_cell_annotations_June2020.rds'
    # new_batch_corrected_data = '/data/leslie/krc3004/RCC_Alireza_Sep2020/ccRCC_6pat_June2020.rds'
    raw_counts_filepath = '/data/leslie/bplee/scBatch/Step0_Data/data/200929_raw_counts.rds'

    # annot = readRDS(annot_filepath)
    # raw_counts = readRDS(raw_counts_filepath) # this list of counts was generated by opening the seurat obj in R and saving the raw counts to a df for saveRDS
    # raw_counts = readRDS(raw_counts_filepath).transpose()
    # test = pyreadr.read_r(raw_counts_filepath)[None].transpose()

    # so this contains 31 labels instead of 16

    #data = readRDS(new_batch_corrected_data)

    

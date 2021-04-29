"""
Contains functions for prepping data from anndata.AnnData objs to DIVALoader objs for training

"""

import anndata
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import torch.utils.data as data_utils


class DIVALoader(data_utils.Dataset):
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


def get_diva_loaders(adata, domain_name="patient", label_name="cell_type", shuffle=False):
    """
    turns adata into two DIVALoader objs, utilizes batch column in adata to separate training and testing data

    Parameters
    ----------
    adata : needs `patients` and `cell_types` columns, 'batch' column should contain 0s (train) and 1s (test)

    Returns
    -------
    counts turn into log counts, then they are divided by the max in each loader to be put between 0 and 1

    """
    if 'log1p' not in adata.uns:
        print("Looks you haven't taken the log of the data, doing it for you")
        sc.pp.log1p(adata)

    train_inds = adata.obs.batch == "0"
    test_inds = ~train_inds
    n_train = sum(train_inds)
    n_test = sum(test_inds)
    data = adata.X
    patients, patient_map = pd.factorize(adata.obs[domain_name])
    labels, label_map = pd.factorize(adata.obs[label_name])
    data_train = data[train_inds,:]
    data_test = data[test_inds,:]
    labels_train = labels[train_inds]
    labels_test = labels[test_inds]
    batch_train = patients[train_inds]
    batch_test = patients[test_inds]

    # doing the normalization thing
    print("normalizing all values between 0 and 1")
    data_train = data_train/np.max(data_train)
    data_test = data_test/np.max(data_test)

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

    train_data_loader, test_data_loader = DIVALoader(), DIVALoader()

    train_data_loader.train = True
    test_data_loader.train = False

    train_data_loader.train_data, test_data_loader.test_data        = data_train.unsqueeze(1), data_test.unsqueeze(1)
    train_data_loader.train_labels, test_data_loader.test_labels    = labels_train, labels_test
    train_data_loader.train_domain, test_data_loader.test_domain    = batch_train, batch_test
    train_data_loader.cell_types, test_data_loader.cell_types       = label_map, label_map
    train_data_loader.patients, test_data_loader.patients           = patient_map, patient_map

    return train_data_loader, test_data_loader



def get_validation_from_training(train_loader_obj, percentage_validation=.1):
    """
    Turns a DIVALoader obj into two, a training and validation set

    Parameters
    ----------
    train_loader_obj
    percentage_validation

    Returns
    -------
    new train loader, validation loader (DIVALoader objs)

    """

    validation_loader = DIVALoader()
    new_train_loader = DIVALoader()
    n = len(train_loader_obj)
    n_valid = int(n*percentage_validation)
    valid_ints = np.random.choice(range(n), n_valid, replace=False)
    train_ints = np.setdiff1d(np.arange(n), valid_ints)

    valid_data, valid_labels, valid_domain = train_loader_obj[valid_ints]
    new_train_data, new_train_labels, new_train_domain = train_loader_obj[train_ints]

    validation_loader.train_data = valid_data
    validation_loader.train_labels = valid_labels
    validation_loader.train_domain = valid_domain
    validation_loader.cell_types = train_loader_obj.cell_types

    new_train_loader.train_data = new_train_data
    new_train_loader.train_labels = new_train_labels
    new_train_loader.train_domain = new_train_domain
    new_train_loader.cell_types = train_loader_obj.cell_types

    return new_train_loader, validation_loader



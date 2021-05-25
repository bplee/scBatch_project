"""
Contains functions for prepping data from anndata.AnnData objs to DIVALoader objs for training

"""

import anndata
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import torch.utils.data as data_utils

from .helper_functions import wrap

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

    data = adata.X
    patients, patient_map = pd.factorize(adata.obs[domain_name])
    labels, label_map = pd.factorize(adata.obs[label_name])

    train_inds = adata.obs.batch == "0"
    n_train = sum(train_inds)
    data_train = data[train_inds, :]
    labels_train = labels[train_inds]
    batch_train = patients[train_inds]

    test_inds = ~train_inds
    n_test = sum(test_inds)
    data_test = data[test_inds,:]
    labels_test = labels[test_inds]
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

    train_data_loader.labels, test_data_loader.labels       = label_map, label_map
    train_data_loader.domains, test_data_loader.domains           = patient_map, patient_map

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
    validation_loader.labels = train_loader_obj.labels
    validation_loader.domains = train_loader_obj.domains

    new_train_loader.train_data = new_train_data
    new_train_loader.train_labels = new_train_labels
    new_train_loader.train_domain = new_train_domain
    new_train_loader.labels = train_loader_obj.labels
    new_train_loader.domains = train_loader_obj.domains

    return new_train_loader, validation_loader


def set_adata_train_test_batches(adata, test, train=None, domain_name="patient"):
    """
    Gives back adata with training ("0") and test ("1") labels specified in adata.obs.batch

    Parameters
    ----------
    adata : anndata.AnnData
    test : list or int
        contains integers corresponding to which labels are going to be test domains
    train : list or int (default: None)
        contains integers corresponding to which labels are going to be train_domains
    domain_name: str (default: "patient")
        name of adata.obs column that contains information that you want to use to stratify domains

    Returns
    -------
    anndata.AnnData
        with added adata.obs.batch column with "0" for training data and "1" for test data

    """
    print(f" Setting training domain: {train}")
    print(f" Setting testing domain: {test}")
    # creating the column
    adata.obs['batch'] = "0"

    # getting the ints:
    domains, domain_map = pd.factorize(adata.obs[domain_name])

    # make sure the type of test and train are lists:
    test = wrap(test)
    # mark all test data
    test_inds = np.isin(domains, test)
    adata.obs.batch[test_inds] = "1"

    if train is None:
        print(f"Test labels: {[domain_map[i] for i in test]}")
        print(f"Train labels: None")
        return adata
    else:
        train = wrap(train)
        train_inds = np.isin(domains, train)
        adata = adata[(train_inds | test_inds),:]
        print(f"Test domains: {[domain_map[i] for i in test]}")
        print(f"Train domains: {[domain_map[i] for i in train]}")
        return adata
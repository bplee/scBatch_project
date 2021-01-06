import os
import sys

import argparse

import torch.utils.data as data_utils

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import anndata
import scanpy as sc

WORKING_DIR = "/data/leslie/bplee/scBatch"

# adding the project dir to the path to import relevant modules below
print("________CHANGING PATH_________")
sys.path.append(WORKING_DIR)
print("\tWorking dir appended to Sys path.")
#from paper_experiments.rotated_mnist.dataset.rcc_loader import RccDataset
from DIVA.dataset.rcc_loader_semi_sup import RccDatasetSemi
from Step0_Data.code.new_data_load import NewRccDatasetSemi as RccDatasetSemi
from Step0_Data.code.starter import get_valid_diva_models

def plot_umap(train_loader, test_loader, model, batch_size, test_patient, train_patient, cell_types, patients):
    model.eval()
    """
    get the latent factors and plot the UMAP plots
    produces 18 plots/testpatient: [zy, zd, zx] x [patient label, cell type label] x [train, test, train+test]
    """
    actuals_d, actuals_y, zy_, zd_, zx_ = [], [], [], [], []

    with torch.no_grad():
        # Train
        patients_train = np.delete(patients, test_patient)
        i = 0
        for (xs, ys, ds) in train_loader:
            i = i + 1
            # To device
            xs, ys, ds = xs.to(device), np.array(ys), np.array(ds)

            # use classification function to compute all predictions for each batch
            zy_loc, zy_scale = model.qzy(xs)
            zd_loc, zd_scale = model.qzd(xs)
            zx_loc, zx_scale = model.qzx(xs)
            zy_.append(np.array(zy_loc.cpu()))
            zd_.append(np.array(zd_loc.cpu()))
            zx_.append(np.array(zx_loc.cpu()))
            actuals_d.append(np.argmax(ds,axis=1))
            actuals_y.append(np.argmax(ys,axis=1))
            if i == 50:
               break

        zy = zy_[0]
        zd = zd_[0]
        zx = zx_[0]
        labels_y = actuals_y[0]
        labels_d = actuals_d[0]
        for i in range(1,50):
            zy = np.vstack((zy, zy_[i]))
            zd = np.vstack((zd, zd_[i]))
            zx = np.vstack((zx, zx_[i]))
            labels_y = np.hstack((labels_y, actuals_y[i]))
            labels_d = np.hstack((labels_d, actuals_d[i]))

        train_only_labels_d = labels_d
        train_only_labels_y = labels_y

        ## Test

        actuals_d, actuals_y, zy_, zd_, zx_ = [], [], [], [], []
        i = 0
        for (xs, ys, ds) in test_loader:
            i = i + 1
            # To device
            xs, ys, ds = xs.to(device), np.array(ys), np.array(ds)

            # use classification function to compute all predictions for each batch
            zy_loc, zy_scale = model.qzy(xs)
            zd_loc, zd_scale = model.qzd(xs)
            zx_loc, zx_scale = model.qzx(xs)
            zy_.append(np.array(zy_loc.cpu()))
            zd_.append(np.array(zd_loc.cpu()))
            zx_.append(np.array(zx_loc.cpu()))
            actuals_d.append(np.argmax(ds, axis=1))
            actuals_y.append(np.argmax(ys, axis=1))
            if i == 50:
                break

        zy = zy_[0]
        zd = zd_[0]
        zx = zx_[0]
        labels_y = actuals_y[0]
        labels_d = actuals_d[0]
        for i in range(1, 50):
            zy = np.vstack((zy, zy_[i]))
            zd = np.vstack((zd, zd_[i]))
            zx = np.vstack((zx, zx_[i]))
            labels_y = np.hstack((labels_y, actuals_y[i]))
            labels_d = np.hstack((labels_d, actuals_d[i]))

        test_only_labels_d = labels_d
        test_only_labels_y = labels_y

        ## Train + Test

        patients = np.append(patients_train, patients[test_patient])
        actuals_d, actuals_y, zy_, zd_, zx_ = [], [], [], [], []
        i = 0
        for (xs, ys, ds) in train_loader:
            i = i + 1
            # To device
            xs, ys, ds = xs.to(device), np.array(ys), np.array(ds)

            # use classification function to compute all predictions for each batch
            zy_loc, zy_scale = model.qzy(xs)
            zd_loc, zd_scale = model.qzd(xs)
            zx_loc, zx_scale = model.qzx(xs)
            zy_.append(np.array(zy_loc.cpu()))
            zd_.append(np.array(zd_loc.cpu()))
            zx_.append(np.array(zx_loc.cpu()))
            actuals_d.append(np.argmax(ds, axis=1))
            actuals_y.append(np.argmax(ys, axis=1))
            if i == 50:
                break

        i = 0
        for (xs, ys, ds) in test_loader:
            i = i + 1
            # To device
            xs, ys, ds = xs.to(device), np.array(ys), np.array(ds)

            # use classification function to compute all predictions for each batch
            zy_loc, zy_scale = model.qzy(xs)
            zd_loc, zd_scale = model.qzd(xs)
            zx_loc, zx_scale = model.qzx(xs)
            zy_.append(np.array(zy_loc.cpu()))
            zd_.append(np.array(zd_loc.cpu()))
            zx_.append(np.array(zx_loc.cpu()))
            actuals_d.append(np.argmax(ds, axis=1))
            actuals_y.append(np.argmax(ys, axis=1))
            if i == 10:
                break

        zy = zy_[0]
        zd = zd_[0]
        zx = zx_[0]
        labels_y = actuals_y[0]
        labels_d = actuals_d[0]
        for i in range(1, 50 + 10):
            zy = np.vstack((zy, zy_[i]))
            zd = np.vstack((zd, zd_[i]))
            zx = np.vstack((zx, zx_[i]))
            labels_y = np.hstack((labels_y, actuals_y[i]))
            labels_d = np.hstack((labels_d, actuals_d[i]))

        train_test_labels_d = labels_d
        train_test_labels_y = labels_y


if __name__ == "__main__":
    # this file is meant to be able to grab all valid models in a dir and run it on all of them

    # function from starter code, returns list of model names (no file ext's), not filepaths
    model_names = get_valid_diva_models()

    # train_patient = 0
    supervised = False
    seed = 0

    # for test_patient in range(6):
    #     if test_patient == train_patient:
    #         continue
    # for model_name in model_names:
    model_name = model_names[0]
    # model_name = './' + 'rcc_new_test_domain_' + str(test_patient) + '_semi_sup_seed_' + str(seed)
    # model_name = f"./rcc_new_test_domain_{test_patient}_train_domain_{train_patient}_semi_sup_seed_{seed}"

    model = torch.load(model_name + '.model')
    args = torch.load(model_name + '.config')
    print(model_name)
    print(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

    try:
        conv = args.conv
    except:
        conv = True

    # Load test
    if ~supervised:
       my_dataset_train = RccDatasetSemi(args.test_patient, args.x_dim, train_patient=args.train_patient, train=True, convolutions=conv)
       my_dataset_test = RccDatasetSemi(args.test_patient, args.x_dim, train_patient=args.train_patient, train=False, convolutions=conv)
       train_loader = data_utils.DataLoader(
                 my_dataset_train,
                 batch_size=args.batch_size,
                 shuffle=True)
       test_loader = data_utils.DataLoader(
                 my_dataset_test,
                 batch_size=args.batch_size,
                 shuffle=True)

    cell_types, patients = my_dataset_train.cell_types_batches()

    # Set seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # plot_umap(train_loader, test_loader, model, args.batch_size, args.test_patient, args.train_patient, cell_types, patients)

    test_patient = args.test_patient
    train_patient = args.train_patient

    actuals_d, actuals_y, zy_, zd_, zx_ = [], [], [], [], []

    with torch.no_grad():
        # Train
        patients_train = np.delete(patients, test_patient)
        i = 0
        for (xs, ys, ds) in train_loader:
            i = i + 1
            # To device
            xs, ys, ds = xs.to(device), np.array(ys), np.array(ds)

            # use classification function to compute all predictions for each batch
            zy_loc, zy_scale = model.qzy(xs)
            zd_loc, zd_scale = model.qzd(xs)
            zx_loc, zx_scale = model.qzx(xs)
            zy_.append(np.array(zy_loc.cpu()))
            zd_.append(np.array(zd_loc.cpu()))
            zx_.append(np.array(zx_loc.cpu()))
            actuals_d.append(np.argmax(ds, axis=1))
            actuals_y.append(np.argmax(ys, axis=1))
            if i == 50:
                break

        zy = zy_[0]
        zd = zd_[0]
        zx = zx_[0]
        labels_y = actuals_y[0]
        labels_d = actuals_d[0]
        for i in range(1, 50):
            zy = np.vstack((zy, zy_[i]))
            zd = np.vstack((zd, zd_[i]))
            zx = np.vstack((zx, zx_[i]))
            labels_y = np.hstack((labels_y, actuals_y[i]))
            labels_d = np.hstack((labels_d, actuals_d[i]))

        train_only_labels_d = labels_d
        train_only_labels_y = labels_y

        ## Test

        actuals_d, actuals_y, zy_, zd_, zx_ = [], [], [], [], []
        i = 0
        for (xs, ys, ds) in test_loader:
            i = i + 1
            # To device
            xs, ys, ds = xs.to(device), np.array(ys), np.array(ds)

            # use classification function to compute all predictions for each batch
            zy_loc, zy_scale = model.qzy(xs)
            zd_loc, zd_scale = model.qzd(xs)
            zx_loc, zx_scale = model.qzx(xs)
            zy_.append(np.array(zy_loc.cpu()))
            zd_.append(np.array(zd_loc.cpu()))
            zx_.append(np.array(zx_loc.cpu()))
            actuals_d.append(np.argmax(ds, axis=1))
            actuals_y.append(np.argmax(ys, axis=1))
            if i == 50:
                break

        zy = zy_[0]
        zd = zd_[0]
        zx = zx_[0]
        labels_y = actuals_y[0]
        labels_d = actuals_d[0]
        for i in range(1, 50):
            zy = np.vstack((zy, zy_[i]))
            zd = np.vstack((zd, zd_[i]))
            zx = np.vstack((zx, zx_[i]))
            labels_y = np.hstack((labels_y, actuals_y[i]))
            labels_d = np.hstack((labels_d, actuals_d[i]))

        test_only_labels_d = labels_d
        test_only_labels_y = labels_y

        ## Train + Test

        patients = np.append(patients_train, patients[test_patient])
        actuals_d, actuals_y, zy_, zd_, zx_ = [], [], [], [], []
        i = 0
        for (xs, ys, ds) in train_loader:
            i = i + 1
            # To device
            xs, ys, ds = xs.to(device), np.array(ys), np.array(ds)

            # use classification function to compute all predictions for each batch
            zy_loc, zy_scale = model.qzy(xs)
            zd_loc, zd_scale = model.qzd(xs)
            zx_loc, zx_scale = model.qzx(xs)
            zy_.append(np.array(zy_loc.cpu()))
            zd_.append(np.array(zd_loc.cpu()))
            zx_.append(np.array(zx_loc.cpu()))
            actuals_d.append(np.argmax(ds, axis=1))
            actuals_y.append(np.argmax(ys, axis=1))
            if i == 50:
                break

        i = 0
        for (xs, ys, ds) in test_loader:
            i = i + 1
            # To device
            xs, ys, ds = xs.to(device), np.array(ys), np.array(ds)

            # use classification function to compute all predictions for each batch
            zy_loc, zy_scale = model.qzy(xs)
            zd_loc, zd_scale = model.qzd(xs)
            zx_loc, zx_scale = model.qzx(xs)
            zy_.append(np.array(zy_loc.cpu()))
            zd_.append(np.array(zd_loc.cpu()))
            zx_.append(np.array(zx_loc.cpu()))
            actuals_d.append(np.argmax(ds, axis=1))
            actuals_y.append(np.argmax(ys, axis=1))
            if i == 10:
                break

        zy = zy_[0]
        zd = zd_[0]
        zx = zx_[0]
        labels_y = actuals_y[0]
        labels_d = actuals_d[0]
        for i in range(1, 50 + 10):
            zy = np.vstack((zy, zy_[i]))
            zd = np.vstack((zd, zd_[i]))
            zx = np.vstack((zx, zx_[i]))
            labels_y = np.hstack((labels_y, actuals_y[i]))
            labels_d = np.hstack((labels_d, actuals_d[i]))

        train_test_labels_d = labels_d
        train_test_labels_y = labels_y
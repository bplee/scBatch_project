import os
import sys
import argparse
import torch.utils.data as data_utils
import numpy as np
import matplotlib.pyplot as plt
import torch
import anndata
import scanpy as sc

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
print("________CHANGING PATH_________")
sys.path.append(WORKING_DIR)
print("\tWorking dir appended to Sys path.")
from Step0_Data.code.starter import *
from Step6_RCC_to_CRC.code.rcc_to_crc_test import *
from Step6_RCC_to_CRC.code.rcc_to_crc_diva import get_accuracy


# This script is explicitly for the data load process for CRC only

if __name__ == "__main__":
    order = []
    test_accuracy_d_list = []
    test_accuracy_y_list = []
    test_accuracy_y_list_weighted = []
    supervised = 0
    main_dir = os.getcwd()
    out_dir = 'cm_figs'
    # getting the name of the directory
    if main_dir[:5] == "/lila":
        main_dir = main_dir[5:]


    # if the folder to save cm figs to doesn't exist, then create it:
    ensure_dir(out_dir)

    diva_models = get_valid_diva_models()
    for f in diva_models:
        model_name = os.path.join(main_dir, f)
        model = torch.load(model_name + '.model')
        args = torch.load(model_name + '.config')
        print(model_name)
        print(args)
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if args.cuda else "cpu")
        kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
        old_load = False
        shuffle = False
        cell_types_to_remove = ["Plasma"]
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
        crc_adata.obsm["X_umap"] = np.array(load_umap())
        if old_load:
            crc_adata.obs['cell_type'] = load_louvain().cell_types
        else:
            crc_adata.obs['cell_type'] = load_louvain().chirag
        crc_patient = crc_adata.obs.batch
        patients = np.unique(crc_adata.obs.batch)
        crc_adata.obs['patient'] = crc_adata.obs.batch
        crc_adata.obs.batch = "0"
        print(f"Selecting test patient: {args.test_patient} ({patients[args.test_patient]})")
        crc_adata.obs.batch[crc_adata.obs.patient == patients[args.test_patient]] = "1"
        gene_ds = GeneExpressionDataset()
        gene_ds.populate_from_data(X=crc_adata.X,
                                   gene_names=np.array(crc_adata.var.index),
                                   batch_indices=pd.factorize(crc_patient)[0],
                                   remap_attributes=False)
        gene_ds.subsample_genes(784)
        crc_adata = crc_adata[:, gene_ds.gene_names]
        train_loader, test_loader = get_diva_loaders(crc_adata)
        data_loaders = {}
        # Load supervised training
        train_loader_sup = data_utils.DataLoader(
            train_loader,
            batch_size=args.batch_size,
            shuffle=True)
        # Load unsupervised training (test set with no labels)
        train_loader_unsup = data_utils.DataLoader(
            test_loader,
            batch_size=args.batch_size,
            shuffle=True)
        data_loaders['sup'] = train_loader_sup
        data_loaders['unsup'] = train_loader_unsup
        cell_types = train_loader.cell_types
        patients = test_loader.patients
        test_accur_d, test_accur_y, test_accur_y_weighted = get_accuracy(train_loader_unsup, model, device, save=f)
        test_accuracy_d_list.append(test_accur_d)
        test_accuracy_y_list.append(test_accur_y)
        test_accuracy_y_list_weighted.append(test_accur_y_weighted)
        order.append(args.test_patient)

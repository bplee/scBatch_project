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
from Step0_Data.code.new_data

def plot_tsne(train_loader, test_loader, model, batch_size, test_patient, cell_types, patients):
    model.eval()
    """
    get the latent factors and plot the TSNE plots
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

        zy_adata, zd_adata, zx_adata = [anndata.AnnData(_) for _ in [zy, zd, zx]]

        name = ['zy', 'zd', 'zx']
        for i, _ in enumerate([zy_adata, zd_adata, zx_adata]):
            _.obs['batch'] = labels_d
            _.obs['cell_type'] = labels_y
            save_name_pat = '_diva_semi_sup_train_' + name[i] + '_by_batches_heldout_pat_' + str(test_patient) + '.png'
            save_name_cell_type = '_diva_semi_sup_train_' + name[i] + '_by_label_heldout_pat_' + str(test_patient) + '.png'
            sc.pp.neighbors(_, use_rep="X", n_neighbors=15)
            sc.tl.umap(_, min_dist=.3)
            sc.pl.umap(_, color='batch', size=15, alpha=.8, save=save_name_pat)
            sc.pl.umap(_, color='cell_type', size=15, alpha=.8, save=save_name_pat)
        
        # compute the number of accurate predictions
        #zy_tsne = TSNE(n_components=2).fit_transform(zy)
        #zd_tsne = TSNE(n_components=2).fit_transform(zd)
        #zx_tsne = TSNE(n_components=2).fit_transform(zx)

        ## TSNE plots

        #plt.figure(figsize = (20,14))
        #colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
        #for i, cell_type in zip(range(len(cell_types)), cell_types):
        #    if i < 10:
        #       plt.scatter(zy_tsne[labels_y == i, 0], zy_tsne[labels_y == i, 1], c = colors[i], label = cell_type)
        #    else:
        #       plt.scatter(zy_tsne[labels_y == i, 0], zy_tsne[labels_y == i, 1], c = colors[i%10], label = cell_type, marker='x')
        #plt.legend()
        #plt.savefig('./figs_diva/fig_diva_tsne_semi_sup_train_zy_by_labels_heldout_pat_'+str(test_patient)+'.pdf')

        #plt.figure(figsize = (20,14))
        #colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
        #for i, batch in zip(range(len(patients_train)), patients_train):
        #    if i < 10:
        #       plt.scatter(zy_tsne[labels_d == i, 0], zy_tsne[labels_d == i, 1], c = colors[i], label = batch)
        #    else:
        #       plt.scatter(zy_tsne[labels_d == i, 0], zy_tsne[labels_d == i, 1], c = colors[i%10], label = batch, marker='x')
        #plt.legend()
        #plt.savefig('./figs_diva/fig_diva_tsne_semi_sup_train_zy_by_batches_heldout_pat_'+str(test_patient)+'.pdf')

        #plt.figure(figsize = (20,14))
        #colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
        #for i, cell_type in zip(range(len(cell_types)), cell_types):
        #    if i < 10:
        #       plt.scatter(zd_tsne[labels_y == i, 0], zd_tsne[labels_y == i, 1], c = colors[i], label = cell_type)
        #    else:
        #       plt.scatter(zd_tsne[labels_y == i, 0], zd_tsne[labels_y == i, 1], c = colors[i%10], label = cell_type, marker='x')
        #plt.legend()
        #plt.savefig('./figs_diva/fig_diva_tsne_semi_sup_train_zd_by_labels_heldout_pat_'+str(test_patient)+'.pdf')

        #plt.figure(figsize = (20,14))
        #colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
        #for i, batch in zip(range(len(patients_train)), patients_train):
        #    if i < 10:
        #       plt.scatter(zd_tsne[labels_d == i, 0], zd_tsne[labels_d == i, 1], c = colors[i], label = batch)
        #    else:
        #       plt.scatter(zd_tsne[labels_d == i, 0], zd_tsne[labels_d == i, 1], c = colors[i%10], label = batch, marker='x')
        #plt.legend()
        #plt.savefig('./figs_diva/fig_diva_tsne_semi_sup_train_zd_by_batches_heldout_pat_'+str(test_patient)+'.pdf')
        
        #plt.figure(figsize = (20,14))
        #colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))                                                                                                   
        #for i, cell_type in zip(range(len(cell_types)), cell_types):
        #    if i < 10:
        #       plt.scatter(zx_tsne[labels_y == i, 0], zx_tsne[labels_y == i, 1], c = colors[i], label = cell_type)
        #    else:
        #       plt.scatter(zx_tsne[labels_y == i, 0], zx_tsne[labels_y == i, 1], c = colors[i%10], label = cell_type, marker='x')
        #plt.legend()
        #plt.savefig('./figs_diva/fig_diva_tsne_semi_sup_train_zx_by_labels_heldout_pat_'+str(test_patient)+'.pdf')
    
        #plt.figure(figsize = (20,14))
        #colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
        #for i, batch in zip(range(len(patients_train)), patients_train):
        #    if i < 10:
        #       plt.scatter(zx_tsne[labels_d == i, 0], zx_tsne[labels_d == i, 1], c = colors[i], label = batch)
        #    else:
        #       plt.scatter(zx_tsne[labels_d == i, 0], zx_tsne[labels_d == i, 1], c = colors[i%10], label = batch, marker='x')
        #plt.legend()
        #plt.savefig('./figs_diva/fig_diva_tsne_semi_sup_train_zx_by_batches_heldout_pat_'+str(test_patient)+'.pdf')

        ## Test
        
        actuals_d, actuals_y, zy_, zd_, zx_ = [], [], [], [], []
        i = 0
        for (xs, ys, ds) in test_loader:
            i = i + 1
            # To device
            xs, ys, ds= xs.to(device), np.array(ys), np.array(ds)

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

        # compute the number of accurate predictions
        #zy_tsne = TSNE(n_components=2).fit_transform(zy)
        #zd_tsne = TSNE(n_components=2).fit_transform(zd)
        #zx_tsne = TSNE(n_components=2).fit_transform(zx)
        
        zy_adata, zd_adata, zx_adata = [anndata.AnnData(_) for _ in [zy, zd, zx]]

        name = ['zy', 'zd', 'zx'] 
        for i, _ in enumerate([zy_adata, zd_adata, zx_adata]):
            _.obs['batch'] = labels_d
            _.obs['cell_type'] = labels_y
            save_name_pat = '_diva_semi_sup_test_' + name[i] + '_by_batches_heldout_pat_' + str(test_patient) + '.png'
            save_name_cell_type = '_diva_semi_sup_test_' + name[i] + '_by_label_heldout_pat_' + str(test_patient) + '.png'
            sc.pp.neighbors(_, use_rep="X", n_neighbors=15)
            sc.tl.umap(_, min_dist=.3)
            sc.pl.umap(_, color='batch', size=15, alpha=.8, save=save_name_pat)
            sc.pl.umap(_, color='cell_type', size=15, alpha=.8, save=save_name_cell_type)



        #plt.figure(figsize = (20,14))
        #colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
        #for i, cell_type in zip(range(len(cell_types)), cell_types):
        #    if i < 10:
        #       plt.scatter(zy_tsne[labels_y == i, 0], zy_tsne[labels_y == i, 1], c = colors[i], label = cell_type)
        #    else:
        #       plt.scatter(zy_tsne[labels_y == i, 0], zy_tsne[labels_y == i, 1], c = colors[i%10], label = cell_type, marker='x')
        #plt.legend()
        #plt.savefig('./figs_diva/fig_diva_tsne_semi_sup_test_zy_by_labels_heldout_pat_'+str(test_patient)+'.pdf')

        #plt.figure(figsize = (20,14))
        #colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
        #for i, cell_type in zip(range(len(cell_types)), cell_types):
        #    if i < 10:
        #       plt.scatter(zd_tsne[labels_y == i, 0], zd_tsne[labels_y == i, 1], c = colors[i], label = cell_type)
        #    else:
        #       plt.scatter(zd_tsne[labels_y == i, 0], zd_tsne[labels_y == i, 1], c = colors[i%10], label = cell_type, marker='x')
        #plt.legend()
        #plt.savefig('./figs_diva/fig_diva_tsne_semi_sup_test_zd_by_labels_heldout_pat_'+str(test_patient)+'.pdf')

        #plt.figure(figsize = (20,14))
        #colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
        #for i, cell_type in zip(range(len(cell_types)), cell_types):
        #    if i < 10:
        #       plt.scatter(zx_tsne[labels_y == i, 0], zx_tsne[labels_y == i, 1], c = colors[i], label = cell_type)
        #    else:
        #       plt.scatter(zx_tsne[labels_y == i, 0], zx_tsne[labels_y == i, 1], c = colors[i%10], label = cell_type, marker='x')
        #plt.legend()
        #plt.savefig('./figs_diva/fig_diva_tsne_semi_sup_test_zx_by_labels_heldout_pat_'+str(test_patient)+'.pdf')

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
            actuals_d.append(np.argmax(ds,axis=1))
            actuals_y.append(np.argmax(ys,axis=1))
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
            actuals_d.append(np.argmax(ds,axis=1))
            actuals_y.append(np.argmax(ys,axis=1))
            if i == 10:
               break

        zy = zy_[0]
        zd = zd_[0]
        zx = zx_[0]
        labels_y = actuals_y[0]
        labels_d = actuals_d[0]
        for i in range(1,50+10):
            zy = np.vstack((zy, zy_[i]))
            zd = np.vstack((zd, zd_[i]))
            zx = np.vstack((zx, zx_[i]))
            labels_y = np.hstack((labels_y, actuals_y[i]))
            labels_d = np.hstack((labels_d, actuals_d[i]))

        zy_adata, zd_adata, zx_adata = [anndata.AnnData(_) for _ in [zy, zd, zx]]

        name = ['zy', 'zd', 'zx']
        for i, _ in enumerate([zy_adata, zd_adata, zx_adata]):
            _.obs['batch'] = labels_d
            _.obs['cell_type'] = labels_y
            save_name_pat = '_diva_semi_sup_train+test_' + name[i] + '_by_batches_heldout_pat_' + str(test_patient) + '.png'
            save_name_cell_type = '_diva_semi_sup_train+test_' + name[i] + '_by_labels_heldout_pat_' + str(test_patient) + '.png'
            sc.pp.neighbors(_, use_rep="X", n_neighbors=15)
            sc.tl.umap(_, min_dist=.3)
            sc.pl.umap(_, color='batch', size=15, alpha=.8, save=save_name_pat)
            sc.pl.umap(_, color='cell_type', size=15, alpha=.8, save=save_name_cell_type)


        # compute the number of accurate predictions
        #zy_tsne = TSNE(n_components=2).fit_transform(zy)
        #zd_tsne = TSNE(n_components=2).fit_transform(zd)
        #zx_tsne = TSNE(n_components=2).fit_transform(zx)

        ## TSNE plots

        #plt.figure(figsize = (20,14))
        #colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
        #for i, cell_type in zip(range(len(cell_types)), cell_types):
        #    if i < 10:
        #       plt.scatter(zy_tsne[labels_y == i, 0], zy_tsne[labels_y == i, 1], c = colors[i], label = cell_type)
        #    else:
        #       plt.scatter(zy_tsne[labels_y == i, 0], zy_tsne[labels_y == i, 1], c = colors[i%10], label = cell_type, marker='x')
        #plt.legend()
        #plt.savefig('./figs_diva/fig_diva_tsne_semi_sup_train+test_zy_by_labels_heldout_pat_'+str(test_patient)+'.pdf')

        #print('labels_d', labels_d)
        #plt.figure(figsize = (20,14))
        #colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
        #for i, batch in zip(range(len(patients)), patients):
        #    print('patients', patients)
        #    print('i', i)
        #    print('batch', batch)
        #    if i < 10:
        #       plt.scatter(zy_tsne[labels_d == i, 0], zy_tsne[labels_d == i, 1], c = colors[i], label = batch)
        #    else:
        #       plt.scatter(zy_tsne[labels_d == i, 0], zy_tsne[labels_d == i, 1], c = colors[i%10], label = batch, marker='x')
        #plt.legend()
        #plt.savefig('./figs_diva/fig_diva_tsne_semi_sup_train+test_zy_by_batches_heldout_pat_'+str(test_patient)+'.pdf')

        #plt.figure(figsize = (20,14))
        #colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
        #for i, cell_type in zip(range(len(cell_types)), cell_types):
        #    if i < 10:
        #       plt.scatter(zd_tsne[labels_y == i, 0], zd_tsne[labels_y == i, 1], c = colors[i], label = cell_type)
        #    else:
        #       plt.scatter(zd_tsne[labels_y == i, 0], zd_tsne[labels_y == i, 1], c = colors[i%10], label = cell_type, marker='x')
        #plt.legend()
        #plt.savefig('./figs_diva/fig_diva_tsne_semi_sup_train+test_zd_by_labels_heldout_pat_'+str(test_patient)+'.pdf')

        #plt.figure(figsize = (20,14))
        #colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
        #for i, batch in zip(range(len(patients)), patients):
        #    if i < 10:
        #       plt.scatter(zd_tsne[labels_d == i, 0], zd_tsne[labels_d == i, 1], c = colors[i], label = batch)
        #    else:
        #       plt.scatter(zd_tsne[labels_d == i, 0], zd_tsne[labels_d == i, 1], c = colors[i%10], label = batch, marker='x')
        #plt.legend()
        #plt.savefig('./figs_diva/fig_diva_tsne_semi_sup_train+test_zd_by_batches_heldout_pat_'+str(test_patient)+'.pdf')
        
        #plt.figure(figsize = (20,14))
        #colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
        #for i, cell_type in zip(range(len(cell_types)), cell_types):
        #    if i < 10:
        #       plt.scatter(zx_tsne[labels_y == i, 0], zx_tsne[labels_y == i, 1], c = colors[i], label = cell_type)
        #    else:
        #       plt.scatter(zx_tsne[labels_y == i, 0], zx_tsne[labels_y == i, 1], c = colors[i%10], label = cell_type, marker='x')
        #plt.legend()
        #plt.savefig('./figs_diva/fig_diva_tsne_semi_sup_train+test_zx_by_labels_heldout_pat_'+str(test_patient)+'.pdf')

        #plt.figure(figsize = (20,14))
        #colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
        #for i, batch in zip(range(len(patients)), patients):
        #    if i < 10:
        #       plt.scatter(zx_tsne[labels_d == i, 0], zx_tsne[labels_d == i, 1], c = colors[i], label = batch)
        #    else:
        #       plt.scatter(zx_tsne[labels_d == i, 0], zx_tsne[labels_d == i, 1], c = colors[i%10], label = batch, marker='x')
        #plt.legend()
        #plt.savefig('./figs_diva/fig_diva_tsne_semi_sup_train+test_zx_by_batches_heldout_pat_'+str(test_patient)+'.pdf')

 

if __name__ == "__main__":
    #test_patient = 5
    supervised = False
    seed = 0

    for test_patient in [5]:

        model_name = './' + 'rcc_new_test_domain_' + str(test_patient) + '_semi_sup_seed_' + str(seed)

        model = torch.load(model_name + '.model')
        args = torch.load(model_name + '.config')
        print(model_name)
        print(args)

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if args.cuda else "cpu")
        kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
        
        # Load test
        if ~supervised:
           my_dataset_train = RccDatasetSemi(args.test_patient, args.x_dim, train=True)
           my_dataset_test = RccDatasetSemi(args.test_patient, args.x_dim, train=False)
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

        plot_tsne(train_loader, test_loader, model, args.batch_size, test_patient, cell_types, patients)

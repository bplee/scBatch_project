import os
import sys

import argparse

import torch.utils.data as data_utils
import seaborn as sn
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

from DIVA.dataset.rcc_loader import RccDataset
#from ForBrennan.DIVA.dataset.rcc_loader_semi_sup import NewRccDatasetSemi as RccDatasetSemi
from Step0_Data.code.new_data_load import NewRccDatasetSemi as RccDatasetSemi

def get_accuracy(data_loader, classifier_fn, batch_size, test_patient, cell_types, num_labels, model_name):
    # model.eval()
    """
    compute the accuracy over the supervised training set or the testing set
    """
    predictions_d, actuals_d, predictions_y, actuals_y = [], [], [], []

    with torch.no_grad():
        # use the right data loader
        for (xs, ys, ds) in data_loader:
            # To device
            xs, ys, ds = xs.to(device), ys.to(device), ds.to(device)

            # use classification function to compute all predictions for each batch
            pred_d, pred_y = classifier_fn(xs)
            predictions_d.append(pred_d)
            actuals_d.append(ds)
            predictions_y.append(pred_y)
            actuals_y.append(ys)

        ## compute the number of accurate predictions
        #accurate_preds_d = 0
        #for pred, act in zip(predictions_d, actuals_d):
        #    for i in range(pred.size(0)):
        #        v = torch.sum(pred[i] == act[i])
        #        accurate_preds_d += (v.item() == 5)

        ## calculate the accuracy between 0 and 1
        #accuracy_d = (accurate_preds_d * 1.0) / len(data_loader.dataset)

        # compute the number of accurate predictions
        accurate_preds_y = 0
        labels_true = []
        labels_pred = []
        for pred, act in zip(predictions_y, actuals_y):
            for i in range(pred.size(0)):
                v = torch.sum(pred[i] == act[i])
                accurate_preds_y += (v.item() == num_labels)
                labels_pred.append(torch.argmax(pred[i]))
                labels_true.append(torch.argmax(act[i]))

        # calculate the accuracy between 0 and 1
        accuracy_y = (accurate_preds_y * 1.0) / len(data_loader.dataset)

        # true and predicted labels for calculating confusion matrix
        labels_pred = np.array(labels_pred).astype(int)
        labels_true = np.array(labels_true).astype(int)

        cm = confusion_matrix(labels_true, labels_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        accuracy_y_weighted = np.mean(np.diag(cm_norm))
        print(f"cm_norm diag: {np.diag(cm_norm)}")
        print(f"np.mean(np.diag(cm_norm)) {accuracy_y_weighted}")
        print(f"sum/num for cm {np.sum(np.diag(cm_norm))/float(len(cm_norm))}")
        print(f"cm.shape {cm.shape}")
        cm_norm_df = pd.DataFrame(cm_norm,index=cell_types,columns=cell_types)
        plt.figure(figsize = (20,20))
        ax = sn.heatmap(cm_norm_df, cmap="YlGnBu", vmin=0, vmax=1,
                linewidths=.5, annot=True, fmt='4.2f', square = True)
        ax.get_ylim()
        ax.set_ylim(num_labels, 0)
        save_name = f"./cm_figs/cm_{model_name}.png"
        # plt.savefig('./figs_diva/fig_diva_cm_semi_sup_heldout_pat_'+str(test_patient)+'.pdf')
        plt.savefig(save_name)

        return accuracy_y, accuracy_y_weighted


if __name__ == "__main__":
    test_accuracy_y_list = []
    test_accuracy_y_list_weighted = []
    supervised = 0
    vae = 0
    #test_patient = 5
    seed = 0
    scnym_exp = True
    main_dir = '/data/leslie/bplee/scBatch/Step2_DIVA/code/'

    for test_patient in range(6):
        for train_patient in [4]:
            if train_patient == test_patient:
                continue
    #for test_patient in [0,4]:
            if supervised:
               model_name = main_dir + 'rcc_new_test_domain_' + str(test_patient) + '_sup_only_seed_' + str(seed)
            else:
               model_name = main_dir + 'rcc_new_test_domain_' + str(test_patient) + '_semi_sup_seed_' + str(seed)
            if vae:
               model_name = main_dir + 'rcc_vae_test_domain_' + str(test_patient) + '_sup_only_seed_' + str(seed)
            if scnym_exp:
                model_name = f"{main_dir}rcc_new_test_domain_{test_patient}_train_domain_{train_patient}_semi_sup_seed_{seed}"

            model = torch.load(model_name + '.model')
            args = torch.load(model_name + '.config')
            print(model_name)
            print(args)

            args.cuda = not args.no_cuda and torch.cuda.is_available()
            device = torch.device("cuda" if args.cuda else "cpu")
            kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

            # Load test
            if supervised:
                my_dataset = RccDataset(args.test_patient, args.x_dim, train_patient=args.train_patient, train=False)
                test_loader_sup = data_utils.DataLoader(
                         my_dataset,
                         batch_size=args.batch_size,
                         shuffle=True)
                cell_types, _ = my_dataset.cell_types_batches()
            else:
               my_dataset = RccDatasetSemi(args.test_patient, args.x_dim, train_patient=args.train_patient, train=False)
               test_loader_sup = data_utils.DataLoader(
                         my_dataset,
                         batch_size=args.batch_size,
                         shuffle=True)
               cell_types, _ = my_dataset.cell_types_batches()

            # Set seed
            torch.manual_seed(args.seed)
            torch.backends.cudnn.benchmark = False
            np.random.seed(args.seed)

            test_accuracy_y, test_accuracy_y_weighted = get_accuracy(test_loader_sup, model.classifier, args.batch_size, test_patient, cell_types, args.y_dim, args.train_patient, model_name)
            test_accuracy_y_list.append(test_accuracy_y)
            test_accuracy_y_list_weighted.append(test_accuracy_y_weighted)
        print("patient %d" %test_patient)
        print(test_accuracy_y_list)
        print(test_accuracy_y_list_weighted)

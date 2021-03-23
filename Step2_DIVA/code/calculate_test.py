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

#getting starter code
from Step0_Data.code.starter import *

from DIVA.dataset.rcc_loader import RccDataset
#from ForBrennan.DIVA.dataset.rcc_loader_semi_sup import NewRccDatasetSemi as RccDatasetSemi
from Step0_Data.code.new_data_load import NewRccDatasetSemi as RccDatasetSemi

def get_accuracy(data_loader, classifier_fn, batch_size, test_patient, cell_types, num_labels, model_name, device):
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
        # # compute the number of accurate predictions
        # accurate_preds_d = 0
        # for pred, act in zip(predictions_d, actuals_d):
        #    for i in range(pred.size(0)):
        #        v = torch.sum(pred[i] == act[i])
        #        accurate_preds_d += (v.item() == 5)
        #
        # ## calculate the accuracy between 0 and 1
        # accuracy_d = (accurate_preds_d * 1.0) / len(data_loader.dataset)
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
        cm_norm_df = pd.DataFrame(cm_norm,index=cell_types,columns=cell_types)
        plt.figure(figsize = (20,20))
        ax = sn.heatmap(cm_norm_df, cmap="YlGnBu", vmin=0, vmax=1,
                linewidths=.5, annot=True, fmt='4.2f', square = True)
        ax.get_ylim()
        ax.set_ylim(num_labels, 0)
        save_name = f"./cm_figs/cm_{model_name}.png"
        # plt.savefig('./figs_diva/fig_diva_cm_semi_sup_heldout_pat_'+str(test_patient)+'.pdf')
        plt.savefig(save_name)

        print(f"weighted: {accuracy_y_weighted}\nunweighted:{accuracy_y}")

        return accuracy_y, accuracy_y_weighted


if __name__ == "__main__":
    test_accuracy_y_list = []
    test_accuracy_y_list_weighted = []
    supervised = 0
    # vae = 0
    # test_patient = 5
    # seed = 0
    # scnym_exp = True
    # main_dir = '/data/leslie/bplee/scBatch/Step2_DIVA/code/'
    main_dir = os.getcwd()
    
    out_dir = 'cm_figs'
    
    # getting the name of the directory
    if main_dir[:5] == "/lila":
        main_dir = main_dir[5:]

    # if the folder to save cm figs to doesn't exist, then create it:
    ensure_dir(out_dir)
    # if not os.path.exists(out_dir):
    #     print('Directory {out_dir} does not exist. Creating directory in {main_dir}.')
    #     os.makedirs(out_dir)
    
    # Starting loop
    # for test_patient in range(6):
    #     for train_patient in [4]:
    #         if train_patient == test_patient:
    #             continue
    diva_models = get_valid_diva_models()
    for f in diva_models:
    #for test_patient in [0,4]:
            # if supervised:
            #     f = main_dir + 'rcc_new_test_domain_' + str(test_patient) + '_sup_only_seed_' + str(seed)
            # else:
            #     f = main_dir + 'rcc_new_test_domain_' + str(test_patient) + '_semi_sup_seed_' + str(seed)
            # if vae:
            #     f = main_dir + 'rcc_vae_test_domain_' + str(test_patient) + '_sup_only_seed_' + str(seed)
            # if scnym_exp:
            #     f = f"rcc_new_test_domain_{test_patient}_train_domain_{train_patient}_semi_sup_seed_{seed}"

        # f has no directory structure or a file extension
        model_name = os.path.join(main_dir, f)
        model = torch.load(model_name + '.model')
        args = torch.load(model_name + '.config')
        print(model_name)
        print(args)

        # catching any cases where --conv wasnt an arg and setting it to be true
        try:
            conv = args.conv
        except:
            conv = True

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if args.cuda else "cpu")
        kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
        # Load test
        if supervised:
            my_dataset = RccDataset(args.test_patient, args.x_dim, train_patient=args.train_patient, train=False, convolutions=conv)
            test_loader_sup = data_utils.DataLoader(
                     my_dataset,
                     batch_size=args.batch_size,
                     shuffle=True)
            cell_types, _ = my_dataset.cell_types_batches()
        else:
            my_dataset = RccDatasetSemi(args.test_patient, args.x_dim, train_patient=args.train_patient, train=False, convolutions=conv)
            test_loader_sup = data_utils.DataLoader(
                     my_dataset,
                     batch_size=args.batch_size,
                     shuffle=True)
            cell_types, _ = my_dataset.cell_types_batches()

        # Set seed
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

        test_accuracy_y, test_accuracy_y_weighted = get_accuracy(test_loader_sup, model.classifier, args.batch_size, test_patient, cell_types, args.y_dim, f, device)
        test_accuracy_y_list.append(test_accuracy_y)
        test_accuracy_y_list_weighted.append(test_accuracy_y_weighted)
    print(f"Train patient {args.train_patient}")
    print(f"Accuracies: {test_accuracy_y_list}")
    print(f"Weighted Accuracies: {test_accuracy_y_list_weighted}")


# added code for dumb beta grid search
# from calculate_test import *
# bds = ["10.0"]
# bys = ["10.0"]
#
# strs = ["0.1", "0.5", "1.0", "2.0", "5.0", "10.0"]
#
# all_weighted, all_unweighted = [], []
# for i, bd in enumerate(bds):
#     by_weighted, by_unweighted = [], []
#     for j, by in enumerate(bys):
#         if bd == 1 and by == 1:
#             continue
#         else:
#             weighted, unweighted = [], []
#             for test_pat in range(6):
#                 model_name = f"rcc_no_conv_test_domain_{test_pat}_Bd_{bds[i]}_By_{bys[j]}_semi_sup_seed_0"
#                 model = torch.load(model_name + ".model")
#                 args = torch.load(model_name + ".config")
#                 print(model_name)
#                 print(args)
#                 # catching any cases where --conv wasnt an arg and setting it to be true
#                 try:
#                     conv = args.conv
#                 except:
#                     conv = True
#                 args.cuda = not args.no_cuda and torch.cuda.is_available()
#                 device = torch.device("cuda" if args.cuda else "cpu")
#                 kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
#                 my_dataset = RccDatasetSemi(args.test_patient, args.x_dim, train_patient=args.train_patient,
#                                             train=False, convolutions=args.conv)
#                 test_loader_sup = data_utils.DataLoader(
#                     my_dataset,
#                     batch_size=args.batch_size,
#                     shuffle=True)
#                 cell_types, _ = my_dataset.cell_types_batches()
#                 torch.manual_seed(args.seed)
#                 torch.backends.cudnn.benchmark = False
#                 np.random.seed(args.seed)
#                 test_accuracy_y, test_accuracy_y_weighted = get_accuracy(test_loader_sup, model.classifier,
#                                                                          args.batch_size, args.test_patient,
#                                                                          cell_types,
#                                                                          args.y_dim, model_name, device)
#                 unweighted.append(test_accuracy_y)
#                 weighted.append(test_accuracy_y_weighted)
#             print(f"Beta_d: {bd}, Beta_y: {by}\n Weighted Accur: {weighted}\n Unweighted Accur: {unweighted}")
#             by_weighted.append(weighted)
#             by_unweighted.append(unweighted)
#     all_weighted.append(by_weighted)
#     all_unweighted.append(by_unweighted)
#
#
# # added code for alpha grid search
# from calculate_test import *
# ads = ["15000.0"]
# ays = ["8000.0", "15000.0"]
#
# strs = ["1000.0", "2000.0", "4200.0", "8000.0", "15000.0"]
#
# all_weighted, all_unweighted = [], []
# for i, ad in enumerate(ads):
#     ay_weighted, ay_unweighted = [], []
#     for j, ay in enumerate(ays):
#         weighted, unweighted = [], []
#         for test_pat in range(6):
#             model_name = f"rcc_no_conv_test_domain_{test_pat}_ad_{ads[i]}_ay_{ays[j]}_semi_sup_seed_0"
#             model = torch.load(model_name + ".model")
#             args = torch.load(model_name + ".config")
#             print(model_name)
#             print(args)
#             # catching any cases where --conv wasnt an arg and setting it to be true
#             try:
#                 conv = args.conv
#             except:
#                 conv = True
#             args.cuda = not args.no_cuda and torch.cuda.is_available()
#             device = torch.device("cuda" if args.cuda else "cpu")
#             kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
#             my_dataset = RccDatasetSemi(args.test_patient, args.x_dim, train_patient=args.train_patient,
#                                         train=False, convolutions=args.conv)
#             test_loader_sup = data_utils.DataLoader(
#                 my_dataset,
#                 batch_size=args.batch_size,
#                 shuffle=True)
#             cell_types, _ = my_dataset.cell_types_batches()
#             torch.manual_seed(args.seed)
#             torch.backends.cudnn.benchmark = False
#             np.random.seed(args.seed)
#             test_accuracy_y, test_accuracy_y_weighted = get_accuracy(test_loader_sup, model.classifier,
#                                                                      args.batch_size, args.test_patient,
#                                                                      cell_types,
#                                                                      args.y_dim, model_name, device)
#             unweighted.append(test_accuracy_y)
#             weighted.append(test_accuracy_y_weighted)
#         print("_________________________________________________")
#         print(f"alpha_d: {ad}, alpha_y: {ay}\n Weighted Accur: {weighted}\n Unweighted Accur: {unweighted}")
#         print("_________________________________________________")
#         w_str = [str(i) for i in weighted]
#         unw_str = [str(i) for i in unweighted]
#         with open("alpha_grid_search_weighted.csv", "a") as text_file:
#             text_file.write(f"{ad}, {ay}, {','.join(w_str)}\n")
#         with open("alpha_grid_search_unweighted.csv", "a") as text_file:
#             text_file.write(f"{ad}, {ay}, {','.join(unw_str)}\n")
#         ay_weighted.append(weighted)
#         ay_unweighted.append(unweighted)
#     all_weighted.append(ay_weighted)
#     all_unweighted.append(ay_unweighted)
#
#
# # this is code for supervised diva
# weighted, unweighted = [], []
# for test_pat in range(6):
#     model_name = f"rcc_no_conv_test_domain_{test_pat}_sup_seed_0"
#     model = torch.load(model_name + ".model")
#     args = torch.load(model_name + ".config")
#     print(model_name)
#     print(args)
#     # catching any cases where --conv wasnt an arg and setting it to be true
#     try:
#         conv = args.conv
#     except:
#         conv = True
#     args.cuda = not args.no_cuda and torch.cuda.is_available()
#     device = torch.device("cuda" if args.cuda else "cpu")
#     kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
#     my_dataset = RccDatasetSemi(args.test_patient, args.x_dim, train_patient=args.train_patient,
#                                 train=False, convolutions=args.conv)
#     test_loader_sup = data_utils.DataLoader(
#         my_dataset,
#         batch_size=args.batch_size,
#         shuffle=True)
#     cell_types, _ = my_dataset.cell_types_batches()
#     torch.manual_seed(args.seed)
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(args.seed)
#     test_accuracy_y, test_accuracy_y_weighted = get_accuracy(test_loader_sup, model.classifier,
#                                                              args.batch_size, args.test_patient,
#                                                              cell_types,
#                                                              args.y_dim, model_name, device)
#     unweighted.append(test_accuracy_y)
#     weighted.append(test_accuracy_y_weighted)
# print(f"SUPERVISED\nWeighted Accur: {weighted}\n Unweighted Accur: {unweighted}")


# code for the 512 embedding layer
test_accuracy_y_list = []
test_accuracy_y_list_weighted = []
supervised = 0
for i in range(6):
    model_name = f"rcc_no_conv_test_domain_{i}_encoding_dim_512_semi_sup_seed_0"
    model = torch.load(model_name + '.model')
    args = torch.load(model_name + '.config')
    print(model_name)
    print(args)
    # catching any cases where --conv wasnt an arg and setting it to be true
    try:
        conv = args.conv
    except:
        conv = True
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    # Load test
    if supervised:
        my_dataset = RccDataset(args.test_patient, args.x_dim, train_patient=args.train_patient, train=False, convolutions=conv)
        test_loader_sup = data_utils.DataLoader(
                 my_dataset,
                 batch_size=args.batch_size,
                 shuffle=True)
        cell_types, _ = my_dataset.cell_types_batches()
    else:
        my_dataset = RccDatasetSemi(args.test_patient, args.x_dim, train_patient=args.train_patient, train=False, convolutions=conv)
        test_loader_sup = data_utils.DataLoader(
                 my_dataset,
                 batch_size=args.batch_size,
                 shuffle=True)
        cell_types, _ = my_dataset.cell_types_batches()
    # Set seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    test_accuracy_y, test_accuracy_y_weighted = get_accuracy(test_loader_sup, model.classifier, args.batch_size, i, cell_types, args.y_dim, model_name, device)
    test_accuracy_y_list.append(test_accuracy_y)
    test_accuracy_y_list_weighted.append(test_accuracy_y_weighted)
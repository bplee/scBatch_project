import sys
import os
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

WORKING_DIR = "/data/leslie/bplee/scBatch_project"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("CHANGING PATH")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

from Step0_Data.code.starter import *
from TIC_atlas.code.load_data import *


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='svm_TIC')
    # parser.add_argument('--test_patient', nargs="+", default=0,
    #                     help='test domain')
    # parser.add_argument('--train_patient', nargs="+", default=None,
    #                     help='test domain')
    # args_scnym = parser.parse_args()
    # print(args_scnym)
    #
    # print(f"Current Working Dir: {os.getcwd()}")
    #
    # train_pat = args_scnym.train_patient
    # test_pat = args_scnym.test_patient
    train_pat = None

    accurs, w_accurs = [], []
    for test_pat in range(10):
        train_loader, test_loader = load_TIC_diva_datasets(test_domain=test_pat, train_domain=train_pat)
        x = np.array(train_loader.train_data.squeeze(1))
        y = np.array(train_loader.train_labels)@ np.arange(len(train_loader.cell_types))
        test_x = np.array(test_loader.test_data.squeeze(1))
        test_y = np.array(test_loader.test_labels) @ np.arange(len(train_loader.cell_types))
        cell_types = train_loader.cell_types
        svm = LinearSVC()
        svm.fit(x, y)
        train_accur = sum(np.equal(svm.predict(x), y))/len(y)
        test_preds = svm.predict(test_x)
        test_accur = sum(np.equal(test_preds, test_y))/len(test_y)
        cm = confusion_matrix(test_y, test_preds)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        weighted_accuracy = np.mean(np.diag(cm_norm))
        ensure_dir("cm_figs")
        print("Making confusion matrix")
        cm_norm_df = pd.DataFrame(cm_norm, index=cell_types, columns=cell_types)
        plt.figure(figsize=(20, 20))
        ax = sns.heatmap(cm_norm_df, cmap="YlGnBu", vmin=0, vmax=1,
                        linewidths=.5, annot=True, fmt='4.2f', square=True)
        name = f'cm_figs/fig_svm_cm_sup_test_pat_{test_pat}.png'
        plt.savefig(name)
        print(f"Unweighted:\n training accuracy: {train_accur}\n testing accuracy: {test_accur}")
        print(f"Weighted Test Accuracy: {weighted_accuracy}")
        accurs.append(test_accur)
        w_accurs.append(weighted_accuracy)
    print(f"weighted: {w_accurs}")
    print(f"unweighed: {accurs}")
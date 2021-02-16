import sys
import os
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("CHANGING PATH")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

from Step0_Data.code.pkl_load_data import PdRccAllData
from Step0_Data.code.new_data_load import NewRccDatasetSemi as RccDatasetSemi
from Step0_Data.code.starter import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scANVI')
    parser.add_argument('--test_patient', type=int, default=5,
                        help='test domain')
    parser.add_argument('--train_patient', type=int, default=None,
                        help='test domain')
    args_scnym = parser.parse_args()
    print(args_scnym)

    print(f"Current Working Dir: {os.getcwd()}")

    train_pat = args_scnym.train_patient
    test_pat = args_scnym.test_patient

accurs, w_accurs = [],[]
for test_pat in range(6):
    data_obj = RccDatasetSemi(test_pat, 784, ssl=False, starspace=True, libsize_norm=False)
    x = np.array(data_obj.train_data)
    y = data_obj.train_labels.copy()
    test_x = np.array(data_obj.test_data)
    test_y = data_obj.test_labels.copy()
    cell_types = data_obj.cell_types
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
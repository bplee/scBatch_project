import sys
import os
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("CHANGING PATH")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

from Step0_Data.code.pkl_load_data import PdRccAllData
from Step0_Data.code.new_data_load import NewRccDatasetSemi as RccDatasetSemi


if __name__ == "__main__":
    test_pat = 5
    data_obj = RccDatasetSemi(test_pat, 784, starspace = True)

    x = np.array(data_obj.train_data)
    y = data_obj.train_labels.copy()

    test_x = np.array(data_obj.test_data)
    test_y = data_obj.test_labels.copy()

    svm = LinearSVC()
    svm.fit(x, y)
    train_accur = np.equal(svm.predict(x), y)/len(y)

    test_preds = svm.predict(test_x)
    test_accur = np.equal(test_preds, test_y)/len(test_y)

    cm = confusion_matrix(test_y, test_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    weighted_accuracy = np.mean(np.diag(cm_norm))

    print(f"Unweighted:\n training accuracy: {train_accur}\n testing accuracy: {test_accur}")
    print(f"Weighted Test Accuracy: {weighted_accuracy}")
import sys
import os
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import argparse
import scanpy as sc
from scvi.dataset import GeneExpressionDataset
import time

WORKING_DIR = "/data/leslie/bplee/scBatch_project"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("CHANGING PATH")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

from Step0_Data.code.pkl_load_data import PdRccAllData
# from Step0_Data.code.new_data_load import NewRccDatasetSemi as RccDatasetSemi
from broad_rcc.code.load_data import broad_quick_load
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

broad_adata = broad_quick_load()
broad_genes = set(broad_adata.var_names)
broad_adata.obs['patient'] = broad_adata.obs.donor_id.copy()
broad_adata.obs['cell_type'] = broad_adata.obs.FinalCellType.copy()

rcc_obj = PdRccAllData(labels_to_remove=["Ambiguous", "Megakaryocyte", "TAM/TCR (Ambiguos)",])
rcc_patient = rcc_obj.data.patient
rcc_cell_type = rcc_obj.data.cell_type
rcc_raw_counts = rcc_obj.data.drop(["cell_type", "patient"], axis=1)
# these are the ensembl.gene names
rcc_genes = set(rcc_raw_counts.columns.values)

rcc_adata = anndata.AnnData(rcc_raw_counts)
del rcc_raw_counts

rcc_adata.obs['cell_type'] = rcc_cell_type
# rcc_adata.obs['annotations'] = rcc_cell_type
rcc_adata.obs['patient'] = rcc_obj.data.patient

universe = list(broad_genes.intersection(rcc_genes))
rcc_adata = rcc_adata[:, np.array(universe)]
broad_adata = broad_adata[:, np.array(universe)]

rcc_adata.obs['batch'] = "0"
broad_adata.obs['batch'] = "1"
sc.pp.normalize_total(rcc_adata)
sc.pp.normalize_total(broad_adata)
adata = anndata.concat([rcc_adata, broad_adata])

# sc.pp.normalize_total(adata, 1e5)

gene_ds = GeneExpressionDataset()
pats = rcc_adata.obs.patient
gene_ds.populate_from_data(X=adata.X,
                           gene_names=np.array(rcc_adata.var.index),
                           # batch_indices=pd.factorize(pats)[0],
                           remap_attributes=False)
gene_ds.subsample_genes(784)

rcc_adata = rcc_adata[:, gene_ds.gene_names]
broad_adata = broad_adata[:, gene_ds.gene_names]

print("loading data for the svm")
x = rcc_adata.X.copy()
y = pd.factorize(rcc_adata.obs.cell_type)[0]
test_x = broad_adata.X.copy()
test_y = pd.factorize(adata.obs.cell_type)[0][adata.obs.batch == "1"]
cell_types = pd.factorize(adata.obs.cell_type)[1]

del adata

print("running svm")
start_time = time.time()
svm = LinearSVC()
svm.fit(x, y)
print(f"total time: {time.time() - start_time}")
train_accur = sum(np.equal(svm.predict(x), y))/len(y)
test_preds = svm.predict(test_x)
# test_accur = sum(np.equal(test_preds, test_y))/len(test_y)
cm = confusion_matrix(test_y, test_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# weighted_accuracy = np.mean(np.diag(cm_norm))
ensure_dir("cm_figs")
print("Making confusion matrix")
cm_norm_df = pd.DataFrame(cm_norm, index=cell_types, columns=cell_types)
plt.figure(figsize=(60, 60))
ax = sns.heatmap(cm_norm_df, cmap="YlGnBu", vmin=0, vmax=1,
                linewidths=.5, annot=True, fmt='4.2f', square=True)
name = f'cm_figs/cm_broad_transfer.png'
plt.savefig(name)
print(train_accur)
# print(f"Unweighted:\n training accuracy: {train_accur}\n testing accuracy: {test_accur}")
# print(f"Weighted Test Accuracy: {weighted_accuracy}")
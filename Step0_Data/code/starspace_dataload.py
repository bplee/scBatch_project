import os

from IPython import get_ipython
import argparse
import numpy as np
import pandas as pd
from pandas import DataFrame
#import starwrap as sw
import time
import torch
import pyreadr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.metrics.pairwise import cosine_similarity
#from scvi.dataset import GeneExpressionDataset
save_path = '/data/leslie/alireza/scRNAseq_ccRCC/data/ccRCC'

#save_path = '/data/leslie/bplee/scBatch/Step0_data/code'

print("Starting the run")


annot_tam = pyreadr.read_r('/data/leslie/alireza/scRNAseq_ccRCC/data/ccRCC/pat6_tams_annotations.rds')
df_annot_tam = annot_tam[None]
annot_tcell = pyreadr.read_r('/data/leslie/alireza/scRNAseq_ccRCC/data/ccRCC/pat6_tcells_annotations.rds')
df_annot_tcell = annot_tcell[None]
df_annot_all_6_pat = df_annot_tcell.append(df_annot_tam, ignore_index=True)
cell_type_tcell = np.unique(df_annot_tcell.cluster_name.values)
cell_type_tam = np.unique(df_annot_tam.cluster_name.values)
del annot_tam, df_annot_tam, annot_tcell, df_annot_tcell
pandas2ri.activate()
readRDS = robjects.r['readRDS']
rawdata_tam = readRDS('/data/leslie/alireza/scRNAseq_ccRCC/data/ccRCC/pat6_tams_rawcounts.rds').transpose()
rawdata_tcell = readRDS('/data/leslie/alireza/scRNAseq_ccRCC/data/ccRCC/pat6_tcells_rawcounts.rds').transpose()
rawdata_all_6_pat = np.vstack((rawdata_tcell,rawdata_tam))
del rawdata_tam, rawdata_tcell

n_data_all = rawdata_all_6_pat.shape[0]
n_gene_all = rawdata_all_6_pat.shape[1]

cell_types = np.hstack((cell_type_tcell,cell_type_tam))
labels = np.zeros([n_data_all,1])
for i, c in enumerate(cell_types):
    idx = np.where(df_annot_all_6_pat.cluster_name.values == c)[0]
    labels[idx] = i
labels = labels.astype(int)
n_labels = len(np.unique(labels))

patients = np.unique(df_annot_all_6_pat.Sample.values)
batch_indices = np.zeros([n_data_all,1])
for i, b in enumerate(patients):
    idx = np.where(df_annot_all_6_pat.Sample.values == b)[0]
    batch_indices[idx] = i
batch_indices = batch_indices.astype(int)

df = pd.read_csv(os.path.join(save_path, "gene_names.csv"), header=0, index_col=0)
gene_names = pd.Index(df.x.values)
del df

n_each_cell_type = np.zeros(len(cell_types)).astype(int)
for i in range(len(cell_types)):
    n_each_cell_type[i] = np.sum(labels == i)

# gene_dataset = GeneExpressionDataset()
# gene_dataset.populate_from_data(
#     X=rawdata_all_6_pat,
#     batch_indices=batch_indices,
#     labels=labels,
#     gene_names=gene_names,
#     cell_types=cell_types,
#     remap_attributes = False
# )

data_all = rawdata_all_6_pat
del rawdata_all_6_pat
del df_annot_all_6_pat

## delete highly expressed genes:
# gene_sum = np.mean(data_all, axis=0)
# idx_del = np.argsort(-gene_sum)[0:20]
# data_all = np.delete(data_all, idx_del, axis=1)
# gene_names = np.delete(gene_names, idx_del)

idx_batch_train = ~(batch_indices == args_starspace.test_patient).ravel()
idx_batch_test = (batch_indices == args_starspace.test_patient).ravel()

batch_train = batch_indices[idx_batch_train].ravel()
batch_test = batch_indices[idx_batch_test].ravel()

labels_train = labels[idx_batch_train].ravel()
labels_test = labels[idx_batch_test].ravel()

data_train = data_all[idx_batch_train]
data_test = data_all[idx_batch_test]

data_train = np.log(data_train+1)
data_test = np.log(data_test+1)

n_train = len(labels_train)
n_test = len(labels_test)

del data_all

# Shuffle everything one more time
np.random.seed(args_starspace.seed)
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

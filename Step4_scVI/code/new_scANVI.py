import os
import sys

from IPython import get_ipython

save_path = 'data/ccRCC'
n_epochs_all = None

import numpy as np
import pandas as pd
from pandas import DataFrame
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
from scvi.dataset import CsvDataset, GeneExpressionDataset
from scvi.models import SCANVI, VAE
from scvi.inference import UnsupervisedTrainer, JointSemiSupervisedTrainer, SemiSupervisedTrainer

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("CHANGING PATH:")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

from Step0_Data.code.new_data_load import NewRccDatasetSemi as RccDatasetSemi
from Step0_Data.code.starter import ensure_dir

### Annotating a dataset from another datasets 

# annot_tam = pyreadr.read_r('/data/leslie/alireza/scRNAseq_ccRCC/data/ccRCC/pat6_tams_annotations.rds')
# df_annot_tam = annot_tam[None]
# annot_tcell = pyreadr.read_r('/data/leslie/alireza/scRNAseq_ccRCC/data/ccRCC/pat6_tcells_annotations.rds')
# df_annot_tcell = annot_tcell[None]
# df_annot_all_6_pat = df_annot_tcell.append(df_annot_tam, ignore_index=True)
# cell_type_tcell = np.unique(df_annot_tcell.cluster_name.values)
# cell_type_tam = np.unique(df_annot_tam.cluster_name.values)
# del annot_tam, df_annot_tam, annot_tcell, df_annot_tcell
# pandas2ri.activate()
# readRDS = robjects.r['readRDS']
# rawdata_tam = np.array(readRDS('/data/leslie/alireza/scRNAseq_ccRCC/data/ccRCC/pat6_tams_rawcounts.rds'))
# rawdata_tam = rawdata_tam.T
# rawdata_tcell = np.array(readRDS('/data/leslie/alireza/scRNAseq_ccRCC/data/ccRCC/pat6_tcells_rawcounts.rds'))
# rawdata_tcell = rawdata_tcell.T
# rawdata_all_6_pat = np.vstack((rawdata_tcell,rawdata_tam))
# rawdata_all_6_pat = rawdata_all_6_pat.astype(int)
# del rawdata_tam, rawdata_tcell
#
# n_data_all = rawdata_all_6_pat.shape[0]
# n_gene_all = rawdata_all_6_pat.shape[1]
#
# cell_types = np.hstack((cell_type_tcell,cell_type_tam))
# #cell_types = cell_types.reshape([len(cell_types), 1])
# labels = np.zeros([n_data_all,1])
# for i, c in enumerate(cell_types):
#     idx = np.where(df_annot_all_6_pat.cluster_name.values == c)[0]
#     labels[idx] = i
# labels = labels.astype(int)
#
# patients = np.unique(df_annot_all_6_pat.Sample.values)
# batch_indices = np.zeros([n_data_all,1])
# for i, b in enumerate(patients):
#     idx = np.where(df_annot_all_6_pat.Sample.values == b)[0]
#     batch_indices[idx] = i
# batch_indices = batch_indices.astype(int)
#
# df = pd.read_csv("/data/leslie/alireza/scRNAseq_ccRCC/data/ccRCC/gene_names.csv", header=0, index_col=0)
# gene_names = pd.Index(df.x.values)
# del df
#
# n_each_cell_type = np.zeros(len(cell_types)).astype(int)
# for i in range(len(cell_types)):
#     n_each_cell_type[i] = np.sum(labels == i)
#
# gene_dataset = GeneExpressionDataset()
# gene_dataset.populate_from_data(
#              X=rawdata_all_6_pat,
#              batch_indices=batch_indices,
#              labels=labels,
#              gene_names=gene_names,
#              cell_types=cell_types,
#              remap_attributes=False)
# del rawdata_all_6_pat
# del df_annot_all_6_pat
# gene_dataset.subsample_genes(1000)

data_obj = RccDatasetSemi(test_patient=None, x_dim=784, scanvi=True)

gene_dataset = data_obj.GeneExpressionDataset
batch_indices = data_obj.batch_indices
cell_types, patients = data_obj.cell_types_batches()
n_data_all = len(gene_dataset)

test_patient = 0
n_epochs = 100 if n_epochs_all is None else n_epochs_all
scanvi = SCANVI(gene_dataset.nb_genes, gene_dataset.n_batches, gene_dataset.n_labels, n_latent=100, n_layers=1)
trainer = SemiSupervisedTrainer(scanvi, gene_dataset, seed=0, frequency=1)
trainer.labelled_set = trainer.create_posterior(indices=~(batch_indices == test_patient).ravel())
trainer.labelled_set.to_monitor = ['reconstruction_error', 'accuracy']
trainer.unlabelled_set = trainer.create_posterior(indices=(batch_indices == test_patient).ravel())
trainer.unlabelled_set.to_monitor = ['reconstruction_error', 'accuracy']

# The accuracy
t0 = time.time()
trainer.train(n_epochs=n_epochs)
print('Training time of scANVI: {} mins'.format((time.time() - t0)/60))
trainer.unlabelled_set.accuracy()

accuracy_labelled_set = trainer.history["accuracy_labelled_set"]
accuracy_unlabelled_set = trainer.history["accuracy_unlabelled_set"]
x = np.linspace(0,n_epochs,(len(accuracy_labelled_set)))
plt.figure(figsize = (10,10))
plt.plot(x, accuracy_labelled_set, label="accuracy labelled")
plt.plot(x, accuracy_unlabelled_set, label="accuracy unlabelled")
plt.savefig('fig_monitor_accuracy_test_is_pat_'+str(test_patient)+'.png')

# Confusion matrix and its heatmap
labels_true, predicted_labels_scVI = trainer.unlabelled_set.compute_predictions()
cm = confusion_matrix(labels_true, predicted_labels_scVI)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Weighted accuracy of SCANVI is :', np.mean(np.diag(cm_norm)))
print('Unweighted accuracy of SCANVI is :', np.diag(cm).sum()/cm.sum())
cm_norm_df = pd.DataFrame(cm_norm, index=cell_types, columns=cell_types)
plt.figure(figsize = (20,20))
ax = sn.heatmap(cm_norm_df, cmap="YlGnBu", vmin=0, vmax=1,
              linewidths=.5, annot=True, fmt='4.2f', square = True)
ax.get_ylim()
ax.set_ylim(16, 0)
plt.savefig('fig_scanvi_cm_test_is_pat_'+str(test_patient)+'.png')

#n_samples_tsne = 3000
#trainer.full_dataset.show_t_sne(n_samples=n_samples_tsne, color_by='batches and labels', save_name='fig_tsne_plot_test_is_pat_'+str(test_patient)+'.pdf')


# TSNE plot
X_latent_scanvi, batches, labels_latent = trainer.full_dataset.get_latent()
idx_random = np.random.choice(n_data_all, 5000, replace=False)
X_latent_scanvi_sampled = X_latent_scanvi[idx_random]
labels_latent_sampled = labels_latent[idx_random]
batches_sampled = batches[idx_random].ravel()
X_embedded = TSNE(n_components=2).fit_transform(X_latent_scanvi_sampled)
plt.figure(figsize = (20,14))
colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
for i, cell_types in zip(range(gene_dataset.n_labels), gene_dataset.cell_types):
    if i < 10:
        plt.scatter(X_embedded[labels_latent_sampled == i, 0], X_embedded[labels_latent_sampled == i, 1], c = colors[i], label = cell_types)
    else:
        plt.scatter(X_embedded[labels_latent_sampled == i, 0], X_embedded[labels_latent_sampled == i, 1], c = colors[i%10], label = cell_types, marker='x')
plt.legend()
plt.savefig('fig_scanvi_tsne_by_labels_test_is_pat_'+str(test_patient)+'.png')

plt.figure(figsize = (20,14))
colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
for i, batch in zip(range(len(patients)), patients):
    if i < 10:
        plt.scatter(X_embedded[batches_sampled == i, 0], X_embedded[batches_sampled == i, 1], c = colors[i], label = batch)
    else:
        plt.scatter(X_embedded[batches_sampled == i, 0], X_embedded[batches_sampled == i, 1], c = colors[i%10], label = batch, marker='x')
plt.legend()
plt.savefig('fig_scanvi_tsne_by_batches_test_is_pat_'+str(test_patient)+'.png')




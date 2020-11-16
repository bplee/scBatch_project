import os
import sys

save_path = 'data/ccRCC'
n_epochs_all = None

import numpy as np
import pandas as pd
from pandas import DataFrame
import time

import anndata
import scanpy as sc

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

plt.savefig('fig_scanvi_cm_test_is_pat_'+str(test_patient)+'.png')

#n_samples_tsne = 3000
#trainer.full_dataset.show_t_sne(n_samples=n_samples_tsne, color_by='batches and labels', save_name='fig_tsne_plot_test_is_pat_'+str(test_patient)+'.pdf')


# UMAP Plot
X_latent_scanvi, batches, labels_latent = trainer.full_dataset.get_latent()

umap_adata = anndata.AnnData(X_latent_scanvi)
umap_adata.obs['batch'] = [patients[i[0]] for i in gene_dataset.batch_indices]
umap_adata.obs['cell_type'] = gene_dataset.cell_types

sc.pp.neighbors(umap_adata, n_neighbors=30)
sc.tl.umap(umap_adata)
save_name_batch = f"_scANVI_embedding_by_batches_test_pat_{test_patient}.png"
save_name_label = f"_scANVI_embedding_by_label_test_pat_{test_patient}.png"
sc.pl.umap(umap_adata, color='batch', size=10, alpha=.5, save=save_name_batch)
sc.pl.umap(umap_adata, color='label', size=10, alpha=.5, save=save_name_label)


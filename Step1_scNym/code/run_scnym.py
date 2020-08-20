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

# to put data into anndata.Anndata obj for scNym
import anndata

#save_path = '/data/leslie/bplee/scBatch/Step0_data/code'

print("Starting the run")

parser = argparse.ArgumentParser(description='StarSpace')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')
parser.add_argument('--train_mode', type=int, default=0,
                    help='train_mode')
parser.add_argument('--batch_size', type=int, default=2,
                    help='input batch size for training (default: 64)')
parser.add_argument('--maxNegSamples', type=int, default=10,
                    help='maxNegSamples')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--dim', type=int, default=100,
                    help='dim')
parser.add_argument('--margin', type=float, default=0.05,
                    help='margin')
parser.add_argument('--ngrams', type=int, default=1,
                    help='ngrams')
parser.add_argument('--n_gene_sample_train', type=int, default=10000,
                    help='n_gene_sample_train')
parser.add_argument('--n_gene_sample_test', type=int, default=10000,
                    help='n_gene_sample_test')
parser.add_argument('--n_batch_rep', default=5, type=int,
                    help="n_batch_rep")
parser.add_argument('--test_patient', type=int, default=5,
                    help='test domain')
parser.add_argument('--thread', type=int, default=50,
                    help='thread')
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
args_starspace = parser.parse_args()

print(args_starspace)

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

print(data_train.shape)
print(labels_train.shape)
print(batch_train.shape)
print(data_test.shape)
print(labels_test.shape)
print(batch_test.shape)

train_adata = anndata.AnnData(data_train)
target_adata = anndata.AnnData(data_test)

train_adata.obs['annotations'] = labels_train

target_adata.obs['annotations'] = 'Unlabled'

adata = train_adata.concatenate(target_adata)

print("%d cells, %d genes in the joined training and target set" % adata.shape)

# Beginning of scnym tutorial code:

# -*- coding: utf-8 -*-
"""scnym_atlas_transfer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-xEwHXq4INTSyqWo8RMT_pzCMZXNalex

# scNym Cell Type Classification with Cell Atlas References

This notebook trains `scNym` cell type classification models using a relevant cell atlas dataset as training data and an unlabeled dataset from new experiment as target data.

We provide cell atlases for the mouse (Tabula Muris) and rat (Rat Aging Cell Atlas).

We demonstrate scNym by training on young rat cells and predicting on old cells.
Simply change the `UPLOAD_NEW_DATA` variable in the cells below to upload your own experiment instead.

## Install dependencies and import packages
"""

# !pip install tqdm ConfigArgParse numpy torch pandas scanpy matplotlib seaborn mock
# !pip install scnym

# allow tensorboard outputs even though TF2 is installed
# broke the tensorboard/pytorch API
import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import scnym
import torch

# file downloads
import urllib
import json
import os

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from scnym.api import scnym_api

"""## Get links to Cell Atlas datasets"""

# # download a hash table of addresses to pre-formatted cell atlas datasets
# cell_atlas_json_url = 'https://storage.googleapis.com/calico-website-scnym-storage/link_tables/cell_atlas.json'
# urllib.request.urlretrieve(
#     cell_atlas_json_url,
#     './cell_atlas.json'
# )

# with open('./cell_atlas.json', 'r') as f:
#     CELL_ATLASES = json.load(f)

# print('Available Cell Atlases:')
# for k in CELL_ATLASES.keys():
#     print(k)

"""## Download a cell atlas to use as training data"""

# change this variable to use a different atlas as your
# training data set
# ATLAS2USE = 'rat'

# if ATLAS2USE not in CELL_ATLASES.keys():
#     msg = f'{ATLAS2USE} is not available in the cell atlas directory.'
#     raise ValueError(msg)

# if not os.path.exists('train_data.h5ad'):
#     urllib.request.urlretrieve(
#         CELL_ATLASES[ATLAS2USE],
#         'train_data.h5ad',
#     )
# else:
#     print('`train_data.h5ad` is already present.')
#     print('Do you really want to redownload it?')
#     print('If so, run:')
#     print('\t!rm ./train_data.h5ad')
#     print('in a cell below.')
#     print('Then, rerun this cell.')

# train_adata = anndata.read_h5ad('./train_data.h5ad', )
print('%d cells, %d genes in training data set.' % train_adata.shape)


# Removing these downsampling steps for now:
# NOTE: Here we downsample the atlas to avoid memory issues

# MAX_N_CELLS = 30000
# if train_adata.shape[0] > MAX_N_CELLS:
#     print('Downsampling training data to fit in memory.')
#     print('Note: Remove this step if you have high RAM VMs through Colab Pro.')
#     ridx = np.random.choice(train_adata.shape[0], size=MAX_N_CELLS, replace=False)
#     train_adata = train_adata[ridx, :]
#
# # filter rare genes to save on memory
# n_genes = train_adata.shape[1]
# sc.pp.filter_genes(train_adata, min_cells=20)
# n_genes -= train_adata.shape[1]
# print(f'Removed {n_genes} genes.')
# print('%d cells, %d genes in training data set.' % train_adata.shape)
#
# # save genes used in the model
# np.savetxt('./model_genes.csv', train_adata.var_names, fmt='%s')
#
# # temporary
# model_genes = np.loadtxt('./model_genes.csv', dtype='str')

"""## Import a new target data set

Here, we import a target data set that will be used as unlabeled data during training.
We transfer labels from the training data set (e.g. cell atlas) to the target data set in a final prediction step.

This tutorial uses a subset of the rat aging cell atlas as a target data set, but we provide code to upload your own target data set below.
We have found that uploading your data to Google Drive and then importing it to Colab tends to work best.
If you would like to upload your own data, we assume data is located in `/gdrive_root/scnym_data/target_data.h5ad`.
You can change this assumption in the code below to match the location of your data.

If you upload your own dataset, please format it into an `anndata.AnnData` object and normalize counts to `log(CPM + 1)` before using your data with `scNym`.
We also recommend that you filter out cells with low library sizes and genes with few measured cells using standard quality control practices (see [Ilicic et. al. 2016, *Genome Biology*](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0888-1) for details).
We have provided typical threshold values in the code below, but these should be adjusted to match the library size distribution and number of cells in your dataset.

An example of how to format your data is included below.

```python
adata = anndata.AnnData(
  X = X, # [Cells, Genes] scipy.sparse.csr_matrix or numpy.ndarray
  var = var, # [Genes, Features] pd.DataFrame with gene names as the index
  obs = obs, # [Cells, Features] pd.DataFrame with cell barcodes as the index
)

sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e6)
sc.pp.log1p(adata)
sc.pp.filter_cells(adata, min_counts=500)
sc.pp.filter_genes(adata, min_cells=10)
```
"""

# change this variable to `True` if you would
# like to upload your own data
UPLOAD_NEW_DATA = True

# this block uses old cells from the rat aging cell atlas
# as target data and uses only the young cells as training
# data.
# this code will not run if you are uploading your own data
# and changed the variable above.

# if not UPLOAD_NEW_DATA:
#     # set old cells as target data
#     target_adata = train_adata[train_adata.obs['age'] != 'Y', :]
#     # use only young cells are training data
#     train_adata = train_adata[train_adata.obs['age'] == 'Y', :]

# if UPLOAD_NEW_DATA:
#     from google.colab import drive
#
#     # mount google drive to the Colab runtime
#     drive.mount('/gdrive')
#     # define the location of target data in your Google Drive
#     # "My\ Drive" is the root of your google drive
#     TARGET_PATH = '/gdrive/My\ Drive/scnym_data/target_data.h5ad'
#
#     target_adata = anndata.read_h5ad(
#         TARGET_PATH,
#     )

print('%d cells, %d genes in the training data.' % train_adata.shape)
print('%d cells, %d genes in the target data.' % target_adata.shape)

"""## Train an scNym model

Here, we train an scNym model using the MixMatch semi-supervised learning method to transfer lables from the training data set to the target data set.

## Prepare data for training

The scNym API expects a single `anndata.AnnData` object with a column in `AnnData.obs` defining the annotations to learn.
Cells annotated with the special token `"Unlabeled"` will be treated as part of the target dataset.
These cells will be used for semi-supervised and adversarial training.
"""


"""### Train the scNym model

**NOTE:** Training is computationally expensive and many take 1+ hours using the free Colab GPU.
If you'd like to train more models more quickly, consider [connecting Colab to a local runtime with a GPU](https://research.google.com/colaboratory/local-runtimes.html), using [Colab Pro](https://colab.research.google.com/signup?utm_source=faq&utm_medium=link&utm_campaign=why_arent_resources_guaranteed), or downloading this notebook as a Python script and running it on a GPU equipped machine (e.g. in a cluster at your institution).

scNym saves a copy of the best weights determined using early stopping on the validation criterion in `{out_path}/00_best_model_weights.pkl`.
We load the best weights after training is finished to use for prediction on the target dataset.
"""

scnym_api(
    adata=adata,
    task='train',
    groupby='annotations',
    out_path='./scnym_outputs',
    config='no_new_identity',
)

"""## Predict cell types in target data

After training the model, we load the best set of weights selected using early stopping and predict cell types for the target data set.
"""

# !ls scnym_outputs/

scnym_api(
    adata=adata,
    task='predict',
    key_added='scNym',
    config='no_new_identity',
    trained_model='./scnym_outputs'
)

"""## Plot cell type predictions"""


def match_colors(
        adata: anndata.AnnData,
        source_cat: str,
        target_cat: str,
        extend_pal=sns.color_palette('tab20'),
) -> anndata.AnnData:
    '''Match the colors used for common categories across categorical
    variables in a an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Genes] with `source_cat` and `target_cat` in `.obs`
        if source colors not present in `.uns[f"{source_cat}_colors"]`,
        they will be added using `extend_pal`.
    source_cat : str
        source categorical with a set of colors to copy.
    target_cat : str
        target categorical that will adopt colors from `source_cat`.

    Returns
    -------
    adata : anndata.AnnData
        [Cells, Genes]
    '''
    adata.obs[source_cat] = pd.Categorical(adata.obs[source_cat].tolist())
    if f'{source_cat}_colors' not in adata.uns.keys():
        sc.pl._utils.add_colors_for_categorical_sample_annotation(
            adata,
            source_cat,
            palette=extend_pal,
        )

    # define an rgb to hex mapping
    rgb2hex = lambda r, g, b: f'#{r:02x}{g:02x}{b:02x}'

    source_colors = adata.uns[f'{source_cat}_colors']
    source_levels = pd.Categorical(adata.obs[source_cat]).categories

    source_colors = {
        source_levels[i]: source_colors[i] for i in range(len(source_colors))
    }

    target_levels = pd.Categorical(adata.obs[target_cat]).categories
    target_colors = []

    i = 0  # how many keys have we added from the source colors?
    j = 0  # how many colors have we used from the extending palette?
    for target_lev in target_levels:
        if target_lev in source_colors.keys():
            target_colors.append(source_colors[target_lev])
            i += 1
        else:
            idx2get = len(source_colors) + j + 1
            target_colors.append(
                rgb2hex(*(np.array(extend_pal[idx2get % len(extend_pal)]) * 255).astype(np.int))
            )
            j += 1
    adata.uns[f'{target_cat}_colors'] = target_colors

    return adata


target_adata = adata[adata.obs['batch'] == '1', :]

sc.pl.umap(
    target_adata,
    color='cell_ontology_class',
    palette='tab20',
)

target_adata = match_colors(
    adata=target_adata,
    source_cat='cell_ontology_class',
    target_cat='scNym',
)

sc.pl.umap(
    target_adata,
    color='scNym',
)

sc.pl.umap(
    target_adata,
    color='scNym_confidence',
)

"""## Plot model embeddings

`scnym_api` also extracts the activations of the penultimate neural network layer. These activations represent the embedding learned by the scNym model.
"""

sc.pp.neighbors(adata, use_rep='X_scnym', n_neighbors=30)

sc.tl.umap(adata, min_dist=0.3)

sc.pl.umap(adata, color='batch', size=5., alpha=0.2)

sc.pl.umap(adata, color='cell_ontology_class', size=5., alpha=0.2)

"""## Save scNym annotations to locally or to gDrive"""

target_adata.obs.to_csv(
    './annotations.csv'
)

print("MADE IT TO THE END")

# save files locally
# from google.colab import files
#
# files.download('annotations.csv')
#
# # save files to Google Drive
# from google.colab import drive
#
# drive.mount('/gdrive')

# make a directory if not present already
# !mkdir "/gdrive/My Drive/scnym/"
# copy file to gDrive
# !cp annotations.csv "/gdrive/My Drive/scnym/annotations.csv"

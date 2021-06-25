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
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")
from scBatch.main import DIVAObject
from scBatch.dataprep import set_adata_train_test_batches
from pan_cancer_TIMs.code.load_data import *
from broad_rcc.code.scnym.scnym_leslie_to_broad import *
from scBatch.visualization import save_cm

# helper function for encoding bool into ssl arg
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TIM_atlas_DIVA')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--conv', type=bool, default=False,
                        help='run DIVA with convolutional layers? (default: False)')
    # parser.add_argument('--num-supervised', default=1000, type=int,
    #                    help="number of supervised examples, /10 = samples per class")

    # Choose domains
    parser.add_argument('--test_domain', nargs='+', type=int, default=5,
                        help='test domain')
    parser.add_argument('--train_domain', nargs='+', type=int, default=None,
                        help='train domain')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    adata = quick_load()
    adata.obs.MajorCluster = remove_prefixes(adata.obs.MajorCluster)
    adata = filter_cancers(adata)
    label_counts = get_label_counts(adata.obs, "MajorCluster", "cancer")
    cell_types_to_remove = identify_singleton_labels(label_counts)
    adata = filter_cell_types(adata, cell_types_to_remove)
    adata.obs['cell_type'] = adata.obs['MajorCluster'].copy()
    del adata.obs['MajorCluster']
    adata.obs['domain'] = adata.obs['cancer'].copy()

    sc.pp.normalize_total(adata, 1e5)

    gene_ds = GeneExpressionDataset()
    batches = adata.obs.patient
    gene_ds.populate_from_data(X=adata.X,
                               gene_names=np.array(adata.var.index),
                               batch_indices=pd.factorize(batches)[0],
                               remap_attributes=False)
    gene_ds.subsample_genes(784)

    adata = adata[:, gene_ds.gene_names]

    sc.pp.log1p(adata)
    # batches are going to be built off of adata.obs.subtype
    adata = set_adata_train_test_batches(adata,
                                         test=args.test_domain,
                                         train=args.train_domain,
                                         domain_name="cancer")

    x = adata[adata.obs.batch=="0",:].X
    test_x = adata[adata.obs.batch=="1",:].X
    labels, cell_types = pd.factorize(adata.obs.domain)
    y = labels[adata.obs.batch =="0"]
    test_y = labels[adata.obs.batch=="1"]

    print("running svm")
    start_time = time.time()
    svm = LinearSVC()
    svm.fit(x, y)
    print(f"total time: {time.time() - start_time}")
    train_accur = sum(np.equal(svm.predict(x), y)) / len(y)
    test_preds = svm.predict(test_x)

    from scBatch.visualization import save_cm

    save_cm(labels[test_y], labels[test_preds], name=f"svm_myeloid_cancer_{args.test_domain}", sort_labels=True)

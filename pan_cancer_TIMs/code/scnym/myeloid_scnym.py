import sys
import argparse
import pandas as pd
import numpy as np
import scanpy as sc
import torch
from scvi.dataset import GeneExpressionDataset
from scnym.api import scnym_api

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
    parser = argparse.ArgumentParser(description='TIM_atlas_scnym')

    # Choose domains
    parser.add_argument('--test_domain', nargs='+', type=int, default=5,
                        help='test domain')
    parser.add_argument('--train_domain', nargs='+', type=int, default=None,
                        help='train domain')
    # data loading args
    # parser.add_argument('--clean_data', type=bool, default=True,
    #                     help='gets rid of any labels that arent shared by every patient')
    # dont have an arg for getting rid of certian types

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

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
    adata.obs['annotations'] = adata.obs.cell_type.copy()
    adata.obs['annotations'][adata.obs.batch=="1"] = "Unlabeled"

    domains = np.unique(adata.obs.cancer)

    a = {}
    count = 0
    for i in range(len(domains)):
        if i == args.test_domain:
            a[domains[i]] = 'target_0'
        else:
            a[domains[i]] = f"train_{count}"
            count += 1

    adata.obs['domain_label'] = np.array(list(map(lambda x: a[x], np.array(adata.obs.domain))))

    outpath = f"210624_multi_domain_test_cancer_{args.test_domain}"

    scnym_api(adata=adata, task='train', groupby='annotations',
              domain_groupby='domain_label', out_path= outpath,
              config='no_new_identity')

    predict_from_scnym_model(adata, trained_model=outpath)

    accur, weighted_accur = get_accuracies(adata)
    print(f'{outpath} training results')
    print(f"Weighted Accur: {weighted_accur}\nUnweighted Accur: {accur}")

    test_preds = adata.obs.scNym[adata.obs.batch=="1"]
    test_y = adata.obs.cell_type[adata.obs.batch == "1"]

    save_cm(test_preds, test_y, outpath, sort_labels=True)

    plot_scnym_umap(adata, save_name=f"_{outpath}")
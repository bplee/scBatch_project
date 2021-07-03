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
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    # device = torch.device("cuda" if args.cuda else "cpu")
    # kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

    # Set seed
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(args.seed)

    rcc_obj = PdRccAllData(labels_to_remove=["Ambiguous", "Megakaryocyte", "TAM/TCR (Ambiguos)", ])
    rcc_patient = rcc_obj.data.patient
    rcc_cell_type = rcc_obj.data.cell_type
    rcc_raw_counts = rcc_obj.data.drop(["cell_type", "patient"], axis=1)

    # these are the ensembl.gene names
    rcc_genes = set(rcc_raw_counts.columns.values)

    rcc_adata = anndata.AnnData(rcc_raw_counts)
    rcc_adata.obs['cell_type'] = rcc_cell_type
    rcc_adata.obs['annotations'] = rcc_cell_type
    rcc_adata.obs['domain'] = rcc_obj.data.patient
    del rcc_raw_counts
    sc.pp.normalize_total(rcc_adata, 1e5)
    gene_ds = GeneExpressionDataset()
    pats = rcc_adata.obs.domain
    gene_ds.populate_from_data(X=rcc_adata.X,
                               gene_names=np.array(rcc_adata.var.index),
                               batch_indices=pd.factorize(pats)[0],
                               remap_attributes=False)
    gene_ds.subsample_genes(2000)

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

    outpath = f"210703_multidomain_2000_genes_test_cancer_{args.test_domain}"

    scnym_api(adata=adata, task='train', groupby='annotations',
              domain_groupby='domain_label', out_path= outpath,
              config='no_new_identity')

    predict_from_scnym_model(adata, trained_model=outpath)

    accur, weighted_accur = get_accuracies(adata)
    print(f'{outpath} training results')
    print(f"Weighted Accur: {weighted_accur}\nUnweighted Accur: {accur}")

    test_preds = adata.obs.scNym[adata.obs.batch=="1"]
    test_y = adata.obs.cell_type[adata.obs.batch == "1"]

    save_cm(test_y, test_preds, outpath, sort_labels=True, reduce_cm=False)

    sc.pp.neighbors(adata, use_rep='X_scnym', n_neighbors=30)
    sc.tl.umap(adata, min_dist=.3)
    # save_name = f"_scnym_train_domain_{test_pat}_test_domain_{train_pat}_batches+celltype.png"
    sc.pl.umap(adata, color=['batch', 'domain', 'cell_type'], size=5, alpha=.2, save=f'_{outpath}.png')
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from scvi.dataset import GeneExpressionDataset

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")
from scBatch.main import DIVAModel
from scBatch.dataprep import set_adata_train_test_batches
from load_data import quick_load

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TIC_Atlas_DIVA')
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
    parser.add_argument('--test_patient', nargs='+', type=int, default=5,
                        help='test domain')
    parser.add_argument('--train_patient', nargs='+', type=int, default=None,
                        help='train domain')
    # data loading args
    # parser.add_argument('--clean_data', type=bool, default=True,
    #                     help='gets rid of any labels that arent shared by every patient')
    # dont have an arg for getting rid of certian types

    # Model
    parser.add_argument('--d-dim', type=int, default=12,
                        help='number of classes')
    parser.add_argument('--x-dim', type=int, default=784,
                        help='input size after flattening')
    parser.add_argument('--y-dim', type=int, default=26,  # was 16 for old data
                        help='number of classes')
    parser.add_argument('--zd-dim', type=int, default=64,
                        help='size of latent space 1')
    parser.add_argument('--zx-dim', type=int, default=64,
                        help='size of latent space 2')
    parser.add_argument('--zy-dim', type=int, default=64,
                        help='size of latent space 3')
    parser.add_argument('--encoding-dim', type=int, default=512,
                        help='dimension encoding layers work down to')

    # Aux multipliers
    parser.add_argument('--aux_loss_multiplier_y', type=float, default=4200.,
                        help='multiplier for y classifier')
    parser.add_argument('--aux_loss_multiplier_d', type=float, default=2000.,
                        help='multiplier for d classifier')
    # Beta VAE part
    parser.add_argument('--beta_d', type=float, default=1.,
                        help='multiplier for KL d')
    parser.add_argument('--beta_x', type=float, default=1.,
                        help='multiplier for KL x')
    parser.add_argument('--beta_y', type=float, default=1.,
                        help='multiplier for KL y')

    parser.add_argument('-w', '--warmup', type=int, default=50, metavar='N',
                        help='number of epochs for warm-up. Set to 0 to turn warmup off.')
    parser.add_argument('--max_beta', type=float, default=1., metavar='MB',
                        help='max beta for warm-up')
    parser.add_argument('--min_beta', type=float, default=0.0, metavar='MB',
                        help='min beta for warm-up')

    parser.add_argument('--outpath', type=str, default='./',
                        help='where to save')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

    # Model name
    print(args.outpath)
    model_name = f"{args.outpath}210517_diva_test_pat_{args.test_patient}_train_pat_{args.train_patient}"
    fig_name = f"210517_diva_test_pat_{args.test_patient}_train_pat_{args.train_patient}"
    print(model_name)

    # Choose training domains

    # print('test domain: '+str(args.test_patient))

    # Set seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    adata = quick_load()

    gene_ds = GeneExpressionDataset()
    batches = adata.obs.patient
    gene_ds.populate_from_data(X=adata.X,
                               gene_names=np.array(adata.var.index),
                               batch_indices=pd.factorize(batches)[0],
                               remap_attributes=False)
    gene_ds.subsample_genes(784)

    adata = adata[:, gene_ds.gene_names]
    # batches are going to be built off of adata.obs.subtype
    adata = set_adata_train_test_batches(adata,
                                         test=args.test_patient,
                                         train=args.train_patient,
                                         domain_name="patient")

    adata.obs['patient'] = adata.obs['MajorCluster']
    del adata.obs['MajorCluster']

    diva_obj = DIVAModel(args)
    diva_obj.fit(adata, model_name=f"210517_scBatch_TIMs_test_pat_{args.test_patient}")

import argparse

parser = argparse.ArgumentParser(description='DIVA_args')
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
# data loading args
# parser.add_argument('--clean_data', type=bool, default=True,
#                     help='gets rid of any labels that arent shared by every patient')
# dont have an arg for getting rid of certian types

# Model
parser.add_argument('--d-dim', type=int, default=12,
                    help='number of classes')
parser.add_argument('--x-dim', type=int, default=784,
                    help='input size after flattening')
parser.add_argument('--y-dim', type=int, default=26, # was 16 for old data
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
# parser.add_argument('--outpath', type=str, default='./',
#                     help='where to save')

default_args = parser.parse_args()

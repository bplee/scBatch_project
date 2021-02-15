import argparse
import time
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("________CHANGING PATH_________")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

from ForBrennan.DIVA.model.model_diva_no_convolutions import DIVA
#from DIVA.dataset.rcc_loader_semi_sup import RccDatasetSemi
from Step0_Data.code.new_data_load import NewRccDatasetSemi as RccDatasetSemi

def train(train_loader, model, optimizer, epoch):
    model.train()
    train_loss = 0
    epoch_class_y_loss = 0

    for batch_idx, (x, y, d) in enumerate(train_loader):
        # To device
        x, y, d = x.to(device), y.to(device), d.to(device)

        optimizer.zero_grad()
        loss, class_y_loss = model.loss_function(d, x, y)
        loss.backward()
        optimizer.step()

        train_loss += loss
        epoch_class_y_loss += class_y_loss

    train_loss /= len(train_loader.dataset)
    epoch_class_y_loss /= len(train_loader.dataset)

    return train_loss, epoch_class_y_loss


def get_accuracy(data_loader, classifier_fn, batch_size):
    model.eval()
    """
    compute the accuracy over the supervised training set or the testing set
    """
    predictions_d, actuals_d, predictions_y, actuals_y = [], [], [], []

    with torch.no_grad():
        # use the right data loader
        for (xs, ys, ds) in data_loader:

            # To device
            xs, ys, ds = xs.to(device), ys.to(device), ds.to(device)

            # use classification function to compute all predictions for each batch
            pred_d, pred_y = classifier_fn(xs)
            predictions_d.append(pred_d)
            actuals_d.append(ds)
            predictions_y.append(pred_y)
            actuals_y.append(ys)

        # compute the number of accurate predictions
        accurate_preds_d = 0
        for pred, act in zip(predictions_d, actuals_d):
            for i in range(pred.size(0)):
                v = torch.sum(pred[i] == act[i])
                accurate_preds_d += (v.item() == 5)

        # calculate the accuracy between 0 and 1
        accuracy_d = (accurate_preds_d * 1.0) / len(data_loader.dataset)

        # compute the number of accurate predictions
        accurate_preds_y = 0
        labels_true = []
        labels_pred = []
        for pred, act in zip(predictions_y, actuals_y):
            for i in range(pred.size(0)):
                v = torch.sum(pred[i] == act[i])
                accurate_preds_y += (v.item() == 16)
                labels_pred.append(torch.argmax(pred[i]))
                labels_true.append(torch.argmax(act[i]))

        # calculate the accuracy between 0 and 1
        accuracy_y = (accurate_preds_y * 1.0) / len(data_loader.dataset)

        # true and predicted labels for calculating confusion matrix
        labels_pred = np.array(labels_pred).astype(int)
        labels_true = np.array(labels_true).astype(int)

        cm = confusion_matrix(labels_true, labels_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        accuracy_y_weighted = np.mean(np.diag(cm_norm))

        return accuracy_d, accuracy_y, accuracy_y_weighted


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='TwoTaskVae')
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
    #parser.add_argument('--num-supervised', default=1000, type=int,
    #                    help="number of supervised examples, /10 = samples per class")

    # Choose domains
    # parser.add_argument('--list_train_domains', type=list, default=['0', '15', '30', '45'],
    #                     help='domains used during training')
    # parser.add_argument('--list_test_domain', type=str, default='75',
    #                     help='domain used during testing')
    parser.add_argument('--test_patient', type=int, default=5,
                        help='test domain')

    # Model
    parser.add_argument('--d-dim', type=int, default=5,
                        help='number of classes')
    parser.add_argument('--x-dim', type=int, default=784,
                        help='input size after flattening')
    parser.add_argument('--y-dim', type=int, default=16,
                        help='number of classes')
    parser.add_argument('--zd-dim', type=int, default=64,
                        help='size of latent space 1')
    parser.add_argument('--zx-dim', type=int, default=64,
                        help='size of latent space 2')
    parser.add_argument('--zy-dim', type=int, default=64,
                        help='size of latent space 3')

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
    if args.train_patient is not None:
        model_name = f"{args.outpath}rcc_no_conv_test_domain_{args.test_patient}_train_domain_{args.train_patient}_sup_seed_{args.seed}"
    else:
        # model_name = f"{args.outpath}rcc_no_conv_test_domain_{args.test_patient}_train_domain_ALL_semi_sup_seed_{args.seed}"
        model_name = f"{args.outpath}rcc_no_conv_test_domain_{args.test_patient}_sup_seed_{args.seed}"

    print(model_name)

    print('test domain: '+str(args.test_patient))

    # Set seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # Load supervised training
    train_loader = data_utils.DataLoader(
        RccDatasetSemi(args.test_patient, args.x_dim,
                     train=True, ssl=False, convolutions=args.conv),
        batch_size=args.batch_size,
        shuffle=True)

    # setup the VAE
    model = DIVA(args).to(device)

    # setup the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = 1000.
    best_y_acc = 0.

    early_stopping_counter = 1
    max_early_stopping = 100
    t0 = time.time()
    lr = args.lr

    # training loop
    print('\nStart training:', args)
    torch.save(args, model_name + '.config')
    for epoch in range(1, args.epochs + 1):
        beta = min([args.max_beta, args.max_beta * (epoch * 1.) / args.warmup])
        model.beta_d = beta
        model.beta_y = beta
        model.beta_x = beta

        # train
        if epoch > 100:
           lr = .97 * lr
           optimizer = optim.Adam(model.parameters(), lr=lr)
        avg_epoch_losses_sup, avg_epoch_class_y_loss = train(train_loader, model, optimizer, epoch)

        # store the loss and validation/testing accuracies in the logfile
        str_loss_sup = avg_epoch_losses_sup
        str_print = "{} epoch: avg loss {}".format(epoch, str_loss_sup)
        str_print += ", class y loss {}".format(avg_epoch_class_y_loss)

        # this test accuracy is only for logging, this is not used
        # to make any decisions during training
        train_accuracy_d, train_accuracy_y, train_accuracy_y_weighted = get_accuracy(train_loader, model.classifier, args.batch_size)
        str_print += " train accuracy d {}".format(train_accuracy_d)
        str_print += ", y {}".format(train_accuracy_y)
        str_print += ", y_weighted {}".format(train_accuracy_y_weighted)

        print(str_print)

        if train_accuracy_y > best_y_acc:
            early_stopping_counter = 1

            best_y_acc = train_accuracy_y
            best_loss = avg_epoch_class_y_loss

            torch.save(model, model_name + '.model')

        elif train_accuracy_y == best_y_acc:
            if avg_epoch_class_y_loss < best_loss:
                early_stopping_counter = 1

                best_loss = avg_epoch_class_y_loss

                torch.save(model, model_name + '.model')

            else:
                early_stopping_counter += 1
                if early_stopping_counter == max_early_stopping:
                    break

        else:
            early_stopping_counter += 1
            if early_stopping_counter == max_early_stopping:
                break
        print('time passed: {} mins'.format((time.time() - t0)/60))


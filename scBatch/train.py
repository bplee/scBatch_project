"""
Contains functions for training diva
"""
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.optim as optim

from .helper_functions import ensure_dir


def train(data_loaders, model, optimizer, device):
    """
    runs the inference algorithm for an epoch
    returns the values of all losses separately on supervised and unsupervised parts

    Parameters
    ----------
    data_loaders
    model
    optimizer
    periodic_interval_batches
    device

    Returns
    -------

    """
    model.train()

    # compute number of batches for an epoch
    sup_batches = len(data_loaders["sup"])
    unsup_batches = len(data_loaders["unsup"])
    batches_per_epoch = sup_batches + unsup_batches
    periodic_interval_batches = int(np.around(sup_batches / unsup_batches))

    # initialize variables to store loss values
    epoch_losses_sup = 0
    epoch_losses_unsup = 0
    epoch_class_y_loss = 0

    # setup the iterators for training data loaders
    sup_iter = iter(data_loaders["sup"])
    unsup_iter = iter(data_loaders["unsup"])

    # count the number of supervised batches seen in this epoch
    ctr_unsup = 0
    ctr_sup = 0
    for i in range(batches_per_epoch):

        # whether this batch is supervised or not
        # is_unsupervised = (i % (periodic_interval_batches + 1) == 0) and ctr_unsup < unsup_batches
        is_unsupervised = (i % (periodic_interval_batches) == 0) and ctr_unsup < unsup_batches

        # extract the corresponding batch
        if is_unsupervised:
            ctr_unsup += 1
            if ctr_unsup > unsup_batches:
                print(f"ctr_unsup > unsup_batches, {ctr_unsup} > {unsup_batches}")
                print(f" i: {i}\n ctr_unsup: {ctr_unsup}\n ctr_sup: {ctr_sup}")
                is_unsupervised = False
            (x, y, d) = next(unsup_iter)

        if not is_unsupervised:
            ctr_sup += 1
            if ctr_sup > sup_batches:
                print(f"ctr_sup > sup_batches, {ctr_sup} > {sup_batches}")
                print(f" i: {i}\n ctr_unsup: {ctr_unsup}\n ctr_sup: {ctr_sup}")
                break
            (x, y, d) = next(sup_iter)

        # To device
        x, y, d = x.to(device), y.to(device), d.to(device)
        # run the inference for each loss with supervised or un-supervised
        # data as arguments
        optimizer.zero_grad()

        if is_unsupervised:
            new_loss = model.loss_function(d, x)
            epoch_losses_unsup += new_loss

        else:
            new_loss, class_y_loss = model.loss_function(d, x, y)
            epoch_losses_sup += new_loss
            epoch_class_y_loss += class_y_loss

        # print(epoch_losses_sup, epoch_losses_unsup)
        new_loss.backward()
        optimizer.step()

    # return the values of all losses
    return epoch_losses_sup, epoch_losses_unsup, epoch_class_y_loss


def get_accuracy(data_loader, model, device, save=None):
    """
    computes accuracy for a dataloader and a model
    has the option to save a confusion matrix of the results

    Parameters
    ----------
    data_loader
    model
    device
    save

    Returns
    -------
    sup d accuracy, y accuracy, weighted y accuracy

    """
    model.eval()
    classifier_fn = model.classifier
    n_labels = len(data_loader.dataset[0][1])
    n_batches = len(data_loader.dataset[0][2])
    labels = data_loader.dataset.labels

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
                accurate_preds_d += (v.item() == n_batches)
        # calculate the accuracy between 0 and 1
        accuracy_d = (accurate_preds_d * 1.0) / len(data_loader.dataset)
        # compute the number of accurate predictions
        accurate_preds_y = 0
        labels_true = []
        labels_pred = []
        for pred, act in zip(predictions_y, actuals_y):
            for i in range(pred.size(0)):
                v = torch.sum(pred[i] == act[i])
                accurate_preds_y += (v.item() == n_labels)
                labels_pred.append(torch.argmax(pred[i]))
                labels_true.append(torch.argmax(act[i]))
        # calculate the accuracy between 0 and 1
        accuracy_y = (accurate_preds_y * 1.0) / len(data_loader.dataset)
        # true and predicted labels for calculating confusion matrix
        labels_pred = np.array(labels_pred).astype(int)
        labels_true = np.array(labels_true).astype(int)
        cm = confusion_matrix(labels_true, labels_pred, labels=np.arange(n_labels))
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        diag = np.diag(cm_norm)
        # removing nans
        diag = diag[~np.isnan(diag)]
        accuracy_y_weighted = np.mean(diag)
        if save is not None:
            cm_norm_df = pd.DataFrame(cm_norm, index=labels, columns=labels)
            plt.figure(figsize=(20, 20))
            ax = sns.heatmap(cm_norm_df, cmap="YlGnBu", vmin=0, vmax=1,
                            linewidths=.5, annot=True, fmt='4.2f', square=True)
            ax.get_ylim()
            ax.set_ylim(n_labels, 0)
            ensure_dir("./cm_figs")
            save_name = f"./cm_figs/cm_{save}.png"
            plt.savefig(save_name)
        return accuracy_d, accuracy_y, accuracy_y_weighted


def epoch_procedure(model_name, args, model, data_loaders, device):
    # init
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_loss = 1000.
    best_y_acc = 0.

    early_stopping_counter = 1
    max_early_stopping = 100
    t0 = time.time()
    lr = args.lr

    sup_num = len(data_loaders['sup'])
    unsup_num = len(data_loaders['unsup'])

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
        epoch_losses_sup, epoch_losses_unsup, epoch_class_y_loss = train(data_loaders,
                                                                         model,
                                                                         optimizer,
                                                                         device)

        # compute average epoch losses i.e. losses per example
        avg_epoch_losses_sup = epoch_losses_sup / sup_num
        avg_epoch_losses_unsup = epoch_losses_unsup / unsup_num
        avg_epoch_class_y_loss = epoch_class_y_loss / sup_num

        # store the loss and validation/testing accuracies in the logfile
        print(f"{epoch} epoch: avg losses {avg_epoch_losses_sup} {avg_epoch_losses_unsup}, class y loss {avg_epoch_class_y_loss}")

        # looking at new losses and accuracies on the validation set:
        valid_accur_d, valid_accur_y, valid_accur_y_weighted = get_accuracy(data_loaders["valid"], model, device)
        print(f" Valid accuracy y: {valid_accur_y}\tValid accuracy y weighted {valid_accur_y_weighted}\tValid accuracy d {valid_accur_d}\t")

        if valid_accur_y > best_y_acc:
            early_stopping_counter = 1

            best_y_acc = valid_accur_y
            best_loss = avg_epoch_class_y_loss
            print(f"Saving Model, epoch: {epoch}")
            torch.save(model, model_name + '.model')

        elif valid_accur_y == best_y_acc:
            if avg_epoch_class_y_loss < best_loss:
                early_stopping_counter = 1
                best_loss = avg_epoch_class_y_loss
                torch.save(model, model_name + '.model')

            else:
                early_stopping_counter += 1
                if early_stopping_counter == max_early_stopping:
                    print(f"Early Stopping reached max counter: {max_early_stopping}")
                    break

        else:
            early_stopping_counter += 1
            if early_stopping_counter == max_early_stopping:
                print(f"Early Stopping reached max counter: {max_early_stopping}")
                break
        print(f"time passed: {(time.time() - t0)/60} mins")
    print("Completed Training")
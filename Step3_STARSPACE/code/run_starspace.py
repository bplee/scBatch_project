import os
try:
    # changing dir to get to allow relative filepaths in the python file work
	os.chdir("/data/leslie/bplee/scBatch/ccRCC")
	print(os.getcwd())
except:
	pass

from IPython import get_ipython
import argparse
import numpy as np
import pandas as pd
from pandas import DataFrame
import starwrap as sw
import time
import torch
import pyreadr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.metrics.pairwise import cosine_similarity
#from scvi.dataset import GeneExpressionDataset
save_path = '/data/leslie/alireza/scRNAseq_ccRCC/data/ccRCC'

if __name__ == "__main__":
    # Training settings
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

    print("Done loading data (line 159 run_starspace.py")

    filepath_to_train = "../train_files/trainfile_starspace_test_pat_"+str(args_starspace.test_patient)+".txt"
    print(os.getcwd())
    with open(filepath_to_train, 'w+') as fw:
        for i in range(0,n_train):
            data_train_norm = data_train[i]/data_train[i].sum()
            sample_train = np.random.choice(len(gene_names),
                        args_starspace.n_gene_sample_train, p = data_train_norm)
            sample_train = sample_train.astype(int)
            gene_names_selected = list(gene_names[sample_train])
            line = ["{}".format(g) for g in gene_names_selected]
            line.append("__label__"+str(labels_train[i]))
            line = ["batch_"+str(batch_train[i])] * args_starspace.n_batch_rep + line
            line = ' '.join(line)
            fw.write(line)
            fw.write('\n')

    filepath_to_test = "../train_files/testfile_starspace_test_pat_"+str(args_starspace.test_patient)+".txt"
    with open(filepath_to_test, 'w') as fw:
        for i in range(0,n_test):
            data_test_norm = data_test[i]/data_test[i].sum()
            sample_test = np.random.choice(len(gene_names), 
                        args_starspace.n_gene_sample_test, p = data_test_norm)
            sample_test = sample_test.astype(int)
            gene_names_selected = list(gene_names[sample_test])
            line = ["{}".format(g) for g in gene_names_selected]
            line.append("__label__"+str(labels_test[i]))
            line = ["batch_"+str(batch_test[i])] * args_starspace.n_batch_rep + line
            line = ' '.join(line)
            fw.write(line)
            fw.write('\n')

    # Arguments:
    arg = sw.args()
    arg.trainFile = filepath_to_train
    arg.testFile = filepath_to_test
    arg.predictionFile = "./predictfile_starspace_test_pat_"+str(args_starspace.test_patient)+".txt"
    arg.trainMode = args_starspace.train_mode
    arg.K = 1
    arg.lr = args_starspace.lr
    arg.dim = args_starspace.dim
    arg.epoch = args_starspace.epochs
    arg.margin = args_starspace.margin
    arg.ngrams = args_starspace.ngrams
    arg.batchSize = args_starspace.batch_size
    arg.maxNegSamples = args_starspace.maxNegSamples
    arg.thread = args_starspace.thread

    # Train the model:
    sp = sw.starSpace(arg)
    sp.init()
    print('start training')
    t0 = time.time()
    sp.train()
    print('Training time of StarSpace: {} mins'.format((time.time() - t0)/60)) 

    # Save the model:
    # added in the prior folder path, since were operating out of bplee/scBatch/ccRCC
    # changing folder to  /bplee/scBatch/starspace_models
    os.chdir("/data/leslie/bplee/scBatch/starspace_models")
    
    sp.saveModel('starspace_model_'+str(args_starspace.test_patient))
    sp.saveModelTsv('starspace_model_'+str(args_starspace.test_patient)+'.tsv')

    # Reload the model
    sp.initFromSavedModel('starspace_model_'+str(args_starspace.test_patient))
    sp.initFromTsv('starspace_model_'+str(args_starspace.test_patient)+'.tsv')

    # Test the model and create the file './predictfile_starspace.txt'
    sp.evaluate()
    #sp.nearestNeighbor()

    # Convert TSV model (embeded space) to a matrix
    df = pd.read_csv('starspace_model_'+str(args_starspace.test_patient)+'.tsv', sep='\t', header=None, index_col=0)
    embeded_space = df.values
    embeded_genes = df.index[:-n_labels]

    # Find the vectors of the labels in the embeded space
    labels_vec = np.zeros([n_labels, arg.dim])
    for i in range(n_labels):
        idx = np.where(df.index.values == '__label__'+str(i))[0][0]
        labels_vec[i] = embeded_space[idx]

    # Predict the class
    predict = []
    X_latent_test = np.zeros([n_test, arg.dim])
    with open(arg.testFile) as fp:
        for cnt, line in enumerate(fp):
            #print("Line {}: {}".format(cnt, line))
            a = np.array(sp.getDocVector(line, ' '))
            X_latent_test[cnt] = a
            metric = cosine_similarity(a, labels_vec)
            predict.append(np.argmax(metric))

    top_genes = []
    for i in range(n_labels):
        a = np.reshape(labels_vec[i],(1,arg.dim))
        metric = cosine_similarity(a, embeded_space[0:-n_labels,:]).ravel()
        idx_top_genes = np.argsort(-metric)[0:100]
        top_genes.append(embeded_genes[idx_top_genes])
    top_genes = np.array(top_genes)
    df_top_genes = pd.DataFrame(top_genes)
    df_top_genes.to_csv('top_genes_starspace_test_is_pat_'+str(args_starspace.test_patient)+'.csv')

    # Calculate the accuracy of prediction
    predict_labels = np.array(predict)
    accuracy = np.mean(predict_labels == labels_test)
    print("The prediction accuracy is : ", accuracy)

    # Create the confusion matrix and heatmap
    cm = confusion_matrix(labels_test, predict_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Weighted accuracy of StarSpace is :', np.mean(np.diag(cm_norm)))
    print('Unweighted accuracy of StarSpace is :', np.diag(cm).sum()/cm.sum())
    cm_norm_df = pd.DataFrame(cm_norm,index=cell_types,columns=cell_types)

    #get_ipython().run_line_magic('matplotlib', 'inline')
    plt.figure(figsize = (20,20))
    ax = sn.heatmap(cm_norm_df, cmap="YlGnBu", vmin=0, vmax=1,
                linewidths=.5, annot=True, fmt='4.2f', square = True)
    ax.get_ylim()
    ax.set_ylim(16, 0)
    plt.savefig('fig_starspace_cm_test_is_pat_'+str(args_starspace.test_patient)+'.pdf')

    # TSNE plot
    X_latent_train = np.zeros([n_train, arg.dim])
    with open(arg.trainFile) as fp:
        for cnt, line in enumerate(fp):
            X_latent_train[cnt] = np.array(sp.getDocVector(line, ' '))

    X_starspace = np.vstack((X_latent_train, X_latent_test))
    labels_starspace = np.hstack((labels_train, labels_test))
    batches_starspace = np.hstack((batch_indices[idx_batch_train].ravel(), batch_indices[idx_batch_test].ravel()))
    idx_random = np.random.choice(n_train+n_test, 5000, replace=False)
    X_starspace_sampled = X_starspace[idx_random]
    labels_starspace_sampled = labels_starspace[idx_random]
    batches_starspace_sampled = batches_starspace[idx_random]
    X_embedded = TSNE(n_components=2).fit_transform(X_starspace_sampled)

    plt.figure(figsize = (20,14))
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
    for i, cell_type in zip(range(n_labels), cell_types):
        if i < 10:
            plt.scatter(X_embedded[labels_starspace_sampled == i, 0], X_embedded[labels_starspace_sampled == i, 1], c=colors[i], label=cell_type)
        else:
            plt.scatter(X_embedded[labels_starspace_sampled == i, 0], X_embedded[labels_starspace_sampled == i, 1], c=colors[i%10], label=cell_type, marker='x')
    plt.legend()
    plt.savefig('fig_starspace_tsne_by_labels_test_is_pat_'+str(args_starspace.test_patient)+'.pdf')

    plt.figure(figsize = (20,14))
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
    for i, batch in zip(range(len(patients)), patients):
        if i < 10:
            plt.scatter(X_embedded[batches_starspace_sampled == i, 0], X_embedded[batches_starspace_sampled == i, 1], c=colors[i], label=batch)
        else:
            plt.scatter(X_embedded[batches_starspace_sampled == i, 0], X_embedded[batches_starspace_sampled == i, 1], c=colors[i%10], label=batch, marker='x')
    plt.legend()
    plt.savefig('fig_starspace_tsne_by_batches_test_is_pat_'+str(args_starspace.test_patient)+'.pdf')

    # UMAP plot
    reducer = umap.UMAP()
    umap_embedding = reducer.fit_transform(X_starspace_sampled)

    plt.figure(figsize=(20, 14))
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
    for i, cell_type in zip(range(n_labels), cell_types):
        if i < 10:
            plt.scatter(umap_embedding[labels_starspace_sampled == i, 0], umap_embedding[labels_starspace_sampled == i, 1],
                        c=colors[i], label=cell_type)
        else:
            plt.scatter(umap_embedding[labels_starspace_sampled == i, 0], umap_embedding[labels_starspace_sampled == i, 1],
                        c=colors[i % 10], label=cell_type, marker='x')
    plt.legend()
    plt.savefig('fig_starspace_umap_by_labels_test_is_pat_' + str(args_starspace.test_patient) + '.pdf')

    plt.figure(figsize=(20, 14))
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
    for i, batch in zip(range(len(patients)), patients):
        if i < 10:
            plt.scatter(umap_embedding[batches_starspace_sampled == i, 0], umap_embedding[batches_starspace_sampled == i, 1],
                        c=colors[i], label=batch)
        else:
            plt.scatter(umap_embedding[batches_starspace_sampled == i, 0], umap_embedding[batches_starspace_sampled == i, 1],
                        c=colors[i % 10], label=batch, marker='x')
    plt.legend()
    plt.savefig('fig_starspace_umap_by_batches_test_is_pat_' + str(args_starspace.test_patient) + '.pdf')


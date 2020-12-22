import time
import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata

WORKING_DIR = "/data/leslie/bplee/scBatch"
# adding the project dir to the path to import relevant modules below
if WORKING_DIR not in sys.path:
    print("CHANGING PATH")
    sys.path.append(WORKING_DIR)
    print("\tWorking dir appended to Sys path.")

data_dir = "/data/leslie/bplee/scBatch/CRC_dataset/data/raw_count_files"

def get_pat_id_from_filepath(f):
    return os.path.split(f)[-1].split("_")[0]

def read_data(f):
    """

    Parameters
    ----------
    f : str
        filepath to CRC data file

    Returns
    -------
    pandas df
        contains cluster information, and column for patient id is put in

    """
    rtn = pd.read_csv(f, index_col=0)
    rtn["PATIENT"] = get_pat_id_from_filepath(f)
    return rtn

def concat_data(directory="/data/leslie/bplee/scBatch/CRC_dataset/data/raw_count_files"):
    """

    Parameters
    ----------
    directory : str
        dir where CRC's data is

    Returns
    -------
    pandas df
        contains all counts and patient and cluster columms
    """
    print("Loading CRC Data Files from folder:")
    start_time = time.perf_counter()
    files = os.listdir(directory)
    n = len(files)
    for i, f in enumerate(files):
        print(f"  Completed {i}/{n} files", end='\r')
        df = read_data(os.path.join(directory,f))
        if i == 0:
            rtn = df
        else:
            rtn = pd.concat([rtn, df], axis=0)
    print(f"  Completed {n}/{n} files")
    delta_time = time.perf_counter() - start_time
    print(f"Total Time: {delta_time}")

    # setting all NA vals to zero
    rtn = rtn.fillna(0)

    # reordering to put patients column first
    cols = list(rtn)
    # move the column to head of list using index, pop and insert
    cols.insert(0, cols.pop(cols.index('PATIENT')))
    rtn = rtn[cols]

    return rtn

def save_pd_to_pickle(df, pkl_path="/data/leslie/bplee/scBatch/CRC_dataset/pkl_files/201204_CRC_data.pkl"):
    print("Saving dataframe to pickle")
    df.to_pickle(pkl_path, protocol=4)
    print(f"Saved to {pkl_path}")

def get_ranked_marker_genes(df, patient_name=None):
    """
    perform filtering and leiden clustering of cell count data for a pd

    expects data without batch effects

    Parameters
    ----------
    df : pandas df
        a counts matrix with 2 extra columns for PATIENT and for CLUSTER

    patient_name : str
        (default is None) name to label a figure. If set to None, will not produce a figure

    Returns
    -------
    anndata obj
        with leiden cluster labels, and differential expression for different genes
        see rankings with `pd.DataFrame(adata.uns['rank_genes_groups']['names'])`

    """
    counts = df.drop(['PATIENT', 'CLUSTER'], axis=1)
    pats = np.array(df['PATIENT'])
    clust = np.array(df['CLUSTER']).astype(str)
    adata = anndata.AnnData(counts)
    adata.obs['batch'] = pats
    adata.obs['cluster'] = clust
    adata.obs_names_make_unique()
    # sc.pl.highest_expr_genes(adata, n_top=20, )
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

    # UMAP stuff
    sc.pp.scale(adata, max_value=10)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)

    # leiden clustering
    sc.tl.leiden(adata)

    # saving figure
    save_name = f"_{patient_name}_filtered_leiden.png"
    sc.pl.umap(adata, color=['batch', 'cluster', 'leiden'], save=save_name)

    # Rank genes
    sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
    # sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

    return adata

def get_pval_df(adata):
    """
    Gets df of ranked genes and pval for each group (ranked by "score")

    Parameters
    ----------
    adata : anndata obj
        post ranked marker gene analysis

    Returns
    -------
    pandas df, double column format for each group

    """
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    rtn = pd.DataFrame(
        {group + '_' + key[:1]: result[key][group]
         for group in groups for key in ['names', 'pvals']})
    return rtn

if __name__ == "__main__":
    pkl_path = "/data/leslie/bplee/scBatch/CRC_dataset/pkl_files/201204_CRC_data.pkl"
    # all_data = concat_data()
    all_data = pd.read_pickle(pkl_path)
    patient_subset = ["TS-101T",
                      "TS-104T",
                      "TS-105T",
                      "TS-106T",
                      "TS-108T",
                      "TS-109T",
                      "TS-117T",
                      "TS-122T",
                      "TS-123T",
                      "TS-124T",
                      "TS-125T",
                      "TS-127T",
                      "TS-128T",
                      "TS-129T",
                      "TS-131T",
                      "TS-136T"]
    og_pat_inds = all_data['PATIENT'].isin(patient_subset)
    og_data = all_data[og_pat_inds]

    patient_clusters = []
    for name, df in og_data.groupby('PATIENT'):
        patient_clusters.append(get_ranked_marker_genes(df))
    gene_rank_pds = []
    for i in range(len(patient_clusters)):
        gene_rank_pds.append(get_pval_df(patient_clusters[i]))


    # counts = og_data.drop(['PATIENT','CLUSTER'], axis=1)
    # pats = np.array(og_data['PATIENT'])
    # clust = np.array(og_data['CLUSTER']).astype(str)
    # adata = anndata.AnnData(counts)
    # adata.obs['batch'] = pats
    # adata.obs['cluster'] = clust
    # adata.obs_names_make_unique()
    # # sc.pl.highest_expr_genes(adata, n_top=20, )
    # sc.pp.filter_cells(adata, min_genes=200)
    # sc.pp.filter_genes(adata, min_cells=3)
    # adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    # sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    # adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    # adata = adata[adata.obs.pct_counts_mt < 5, :]
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    # adata.raw = adata
    # adata = adata[:, adata.var.highly_variable]
    # sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    # sc.pp.scale(adata, max_value=10)
    # sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    # sc.tl.umap(adata)
    # sc.pl.umap(adata, color=['batch', 'cluster'])
    # sc.tl.leiden(adata)
    # sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
    # sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
    # sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
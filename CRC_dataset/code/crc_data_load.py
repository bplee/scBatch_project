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

def clean_data_qc(df):
    """
    performs:
        - cell filtering
        - gene filtering
        - removing cells with MT gene count% > 20%
        - saving an adata.raw
        - removing MT and ribo genes
        - normalization
        - log transform

        - umap
        - leiden clustering
        - rank DEG analysis between leiden clusters

        - selecting for highly variable genes PULLS CLUSTERS APART

        - TODO:
            - GENES THAT ARE BROADLY EXPRESSED GET TOSSED OUT
            - LOOK AT MEAN RAW COUNT ACROSS ALL GENES, CUT TAILS
            - DISTRIBUTION OF LIBRARY SIZE AND CUT OFF
            - PLOT LIBRARY CELLS


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

    # identifying mt and ribo genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    adata.var['ribo'] = adata.var_names.str.startswith(("RPS", "RPL"))  # annotate the group of ribosomal genes as 'ribo'
    print(f" Number of MT genes: {sum(adata.var['mt'])} / {adata.shape[0]}")
    print(f" Number of Ribo genes: {sum(adata.var['ribo'])} / {adata.shape[0]}")

    # calculating pct_count_mt/ribo etc.
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # removing all cells with mt% > 20%
    print(f" Number of cells with MT%>20: {sum(adata.obs.pct_counts_mt>20)} ")
    adata = adata[adata.obs.pct_counts_mt<20, :]

    # saving a raw, not sure if its needed anymore, can be access by adata.raw.to_adata()
    adata.raw = adata

    # removing mitochondrial and ribosomal genes:
    print(f" Removing ribo and mitochondrial genes")
    keep_genes = ~adata.var.mt & ~adata.var.ribo
    adata = adata[:, keep_genes]

    # removing sparse genes and cells
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    # adata = adata[adata.obs.pct_counts_mt < 5, :]

    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)


    # performing library size cutoff
    libsizes = adata.X.sum(axis=1)
    log_libsizes = np.log(adata.X.sum(axis=1))
    libsizes = pd.DataFrame({'libsizes' :libsizes, 'log_libsizes': log_libsizes})
    max_cutoff = libsizes.quantile(.95)[0]

    adata = adata[libsizes.libsizes<max_cutoff,:]

    # size_factors = libsizes/np.mean(libsizes)


    # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    # print(f" Number of highly variable MT genes: {sum(adata.var.highly_variable * adata.var.mt)}")
    # print(f" Number of highly variable ribo genes: {sum(adata.var.highly_variable * adata.var.ribo)}")
    # adata = adata[:, adata.var.highly_variable]

    return adata

def get_ranked_marker_genes(adata, patient_name=None):
    """
    Runs the leiden clustering

    Parameters
    ----------
    adata
    patient_name

    Returns
    -------

    """

    # UMAP stuff
    sc.pp.scale(adata, max_value=10)
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=40)
    sc.tl.umap(adata)

    # leiden clustering
    sc.tl.leiden(adata)

    if patient_name is not None:
        # saving figure
        save_name = f"_{patient_name}_filtered_leiden.png"
        sc.pl.umap(adata, color=['batch', 'cluster', 'leiden'], save=save_name)

    # Rank genes
    sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
    if patient_name is not None:
        rank_gene_save_name = f"_pat_{patient_name}.png"
        sc.pl.rank_genes_groups(adata, save_name=rank_gene_save_name)

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

def plot_gene_marker_umaps(adata, gene_markers, cell_type, save_name):
    """
    Plots umap of data colored by the gene markers for a specified cell type

    Needs to be run on the output of `get_ranked_marker_genes`

    Parameters
    ----------
    adata : anndata.AnnData obj
        contains AnnData.obsm["X_umap"] and list of gene names
    gene_markers : pandas df
        dataframe where cols are cell types; each is a list of genes for the cell type

    cell_type : str
        cell type, should be a column in gene_markers
    save_name

    Returns
    -------
    None
        saves umap figs

    """
    marker_gene_names = gene_markers[cell_type].dropna()
    n_gene_markers = len(marker_gene_names)
    markers_in_dataset = marker_gene_names.isin(adata.var.index)
    if sum(markers_in_dataset) < n_gene_markers:
        print(f"The following genes makers for {cell_type} do not exist in the dataset:")
        print(marker_gene_names[~markers_in_dataset])
    umap_colorings = np.array(marker_gene_names[markers_in_dataset])
    umap_colorings = np.append(umap_colorings, "leiden")
    sc.pl.umap(adata, color=umap_colorings, save=save_name)

def assess_marker_genes(df, markers, n_genes=30):
    """
    gets number of marker genes for each cell type (col) in markers that belong to each clusters DEGs

    Parameters
    ----------
    df : pandas df
        of gene names and pvals (return of the `get_pval_df` function)
    markers : pandas df
        of gene markers to search for, where each column is a different cell type
    n_genes : int
        (default is 30) number of top scoring genes to search through for markers

    Returns
    -------
    pandas df [|groups| x |cell types|]

    """

    genes_by_groups = df.iloc[:n_genes, range(0, len(df.columns), 2)]
    # genes_by_groups.iloc[:30, range(0, len(a.columns), 2)].apply(lambda x: sum(markers.T_cell.isin(x)), axis=0)
    return genes_by_groups.apply(lambda y: markers.apply(lambda x: sum(x.isin(y[:n_genes])), axis=0), axis=0)

# def get_vianne_subset(df):
#     """
#     Returns a df of counts for ony the genes listed in viannes file
#
#     Parameters
#     ----------
#     df : pandas df
#         dataframe where column values are gene names that we can subset
#
#     Returns
#     -------
#     subset of the pandas df that was inputted, where you only select specified genes
#
#     """
#     vianne_genes_file_path = '/data/leslie/bplee/scBatch/CRC_dataset/metadata/vianne_gene_subset.txt'
#
#     vianne_genes = pd.read_csv(vianne_genes_file_path).to_numpy().T[0]
#     df.columns

if __name__ == "__main__":
    pkl_path = "/data/leslie/bplee/scBatch/CRC_dataset/pkl_files/201204_CRC_data.pkl"
    # all_data = concat_data()
    all_data = pd.read_pickle(pkl_path)
    # patient_subset = ["TS-101T",
    #                   "TS-104T",
    #                   "TS-105T",
    #                   "TS-106T",
    #                   "TS-108T",
    #                   "TS-109T",
    #                   "TS-117T",
    #                   "TS-122T",
    #                   "TS-123T",
    #                   "TS-124T",
    #                   "TS-125T",
    #                   "TS-127T",
    #                   "TS-128T",
    #                   "TS-129T",
    #                   "TS-131T",
    #                   "TS-136T"]
    patient_subset = ["TS-101T",
                      "TS-104T",
                      "TS-106T",
                      "TS-108T",
                      "TS-109T",
                      "TS-125T"]
    og_pat_inds = all_data['PATIENT'].isin(patient_subset)
    og_data = all_data[og_pat_inds]

    adata = clean_data_qc(og_data)

    # small test data:
    ex_pat = og_data[og_data.PATIENT == 'TS-108T']
    # test = get_ranked_marker_genes(ex_pat)
    # a = get_pval_df(test)

    gene_markers_path = "/data/leslie/bplee/scBatch/CRC_dataset/metadata/immune_markers.xlsx"

    # here columns are the different cell types and rows are diff genes
    # no correspondence between genes in the same row
    first_markers = pd.read_excel(gene_markers_path, sheet_name=0)
    second_markers = pd.read_excel(gene_markers_path, sheet_name=1)

    # # specific example for vianne,
    # vianne = pd.read_csv("../metadata/vianne_gene_subset.txt", header=None)
    # vianne = vianne.to_numpy().T[0]
    # vianne = np.delete(vianne, np.isin(vianne, ['KLRA1', 'CD103', 'CPA1', 'LY6G', 'IL12'])) # missing these cell types
    #
    # df = ex_pat
    #
    # counts = df.drop(['PATIENT', 'CLUSTER'], axis=1)
    # pats = np.array(df['PATIENT'])
    # clust = np.array(df['CLUSTER']).astype(str)
    # adata = anndata.AnnData(counts)
    # adata.obs['batch'] = pats
    # adata.obs['cluster'] = clust
    # adata.obs_names_make_unique()
    # # sc.pl.highest_expr_genes(adata, n_top=20, )
    # sc.pp.filter_cells(adata, min_genes=200)
    # sc.pp.filter_genes(adata, min_cells=3)
    # adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    # adata.var['ribo'] = adata.var_names.str.startswith(
    #     ("RPS", "RPL"))  # annotate the group of ribosomal genes as 'ribo'
    # sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], percent_top=None, log1p=False, inplace=True)
    # print(f" Number of MT genes: {sum(adata.var['mt'])} / {adata.shape[0]}")
    # print(f" Number of Ribo genes: {sum(adata.var['ribo'])} / {adata.shape[0]}")
    # adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    # adata = adata[adata.obs.pct_counts_mt < 5, :]
    # mini = adata[:, vianne]
    # mini.var_names_make_unique()
    # mini.var_names_make_unique()
    # sc.pp.normalize_total(mini, target_sum=1e4)
    # sc.pp.neighbors(mini, n_neighbors=10)
    # sc.tl.umap(mini)
    # label_colors_for_plotting = list(vianne)
    # label_colors_for_plotting.append('total_counts')
    # sc.pl.umap(mini, color=np.array(label_colors_for_plotting), save="_vianne_genes_and_counts.png")


    # patient_clusters = []
    # for name, df in og_data.groupby('PATIENT'):
    #     # getting the ranked marker genes for each patient
    #     patient_clusters.append(get_ranked_marker_genes(df))
    #
    # gene_rank_pds = []
    # for i in range(len(patient_clusters)):
    #     # getting readable tables of top scoring genes and pvals for each group
    #     gene_rank_pds.append(get_pval_df(patient_clusters[i]))
    #
    # n_genes = 30
    # markers_found = []
    # for i, df in enumerate(gene_rank_pds):
    #     markers_found.append(assess_marker_genes(df, first_markers, n_genes=n_genes))

    # batch_corr = pd.read_csv("210121_batch_corrected_data.csv", index_col=0).T
    #
    # og_umi_mapping = {og_data.PATIENT.index.values[i]: og_data.PATIENT.values[i] for i in range(len(og_data.PATIENT))}
    #
    # batch_corrected_patients = []
    # for i, umi in enumerate(umis):
    #     try:
    #         batch_corrected_patients.append(og_umi_mapping[umi])
    #     except:
    #         try:
    #             batch_corrected_patients.append(og_umi_mapping[umi - .1])
    #         except:
    #             batch_corrected_patients.append("Unknown")
    #
    # adata = anndata.AnnData(batch_corr)
    # adata.obs["batch"] = batch_corrected_patients
    # sc.pp.neighbors(adata, n_neigbors=20)
    # sc.tl.umap(adata)
    # sc.tl.leiden(adata)
    # sc.pl.umap(adata, color=["batch", "leiden"], save="_batch_corrected_umap.png")


    # for col in first_markers.columns:
    #     print(col)
    #     for i in range(13):
    #         print(sum(first_markers[col].isin(a[str(i) + '_n'][:30])))
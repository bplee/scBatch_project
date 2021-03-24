# Title     : perform_mnn
# Objective : load data perform mnn using `fastMNN` from batchelor package
# Created by: brennan
# Created on: 1/20/21

library(scran)
library(magrittr)
library(batchelor)
library(scater)
library(scMerge)

# APPLY FUNCTIONS

get_pat_id = function(string){
  filename = tail(strsplit(string, "/")[[1]], n=1)
  rtn = strsplit(filename, "_")[[1]][1]
  return(rtn)
}

read_file = function(file){
  # pat_id = get_pat_id(file)

  # first column contains the UMIs, so we use them as the row name
  rtn = read.csv(file, row.names=1)

  # CLUSTER is the first column, so we take it out:
  cluster = rtn$CLUSTER
  # nothing is being done with the cluster var. for now
  rtn = rtn[-c(1)]

  # Seurat and SingleCellExperiment expects genes to be the rows, cols to be the cells (use transpose)
  rtn = t(rtn)
  rtn = SingleCellExperiment(list(counts=rtn))
  libsizes = colSums(counts(rtn))
  size.factors = libsizes/mean(libsizes)
  logcounts(rtn) = log2(t(t(counts(rtn))/size.factors) + 1)
  return(rtn)
}

read_python_cleaned_data = function (data_file, obs_file){
  rtn = t(read.csv(data_file, row.names=1))
  obs = read.csv(obs_file, row.names=1)
  rtn = SingleCellExperiment(list(counts=rtn))
  colData(rtn)$batch = obs$batch
  libsizes = colSums(counts(rtn))
  size.factors = libsizes/mean(libsizes)
  logcounts(rtn) = log2(t(t(counts(rtn))/size.factors) + 1)
  return(rtn)
}

run_mnn = function(sce_obj){
  out = fastMNN(sce_obj, batch=colData(sce_obj)$batch)
  write.csv(assay(out), "210126_mnn_data.csv")
  return(out)
}

### SCRIPT

#data_dir = "../data/raw_count_files/"
#data_files = paste(data_dir, list.files(data_dir), sep="")
#
#patient_subset = c("TS-101T",
#                   "TS-104T",
#                   "TS-106T",
#                   "TS-108T",
#                   "TS-109T",
#                   "TS-125T")
#
#data_files = paste(data_dir, patient_subset, "_dense.csv", sep="")
#
#pat_ids = sapply(data_files, get_pat_id)
## pat_ids is a vector
#
#all_data = sapply(data_files, read_file)
#names(all_data) = pat_ids
## all data is a list, each element is a dataframe
#
#batch_sce = sce_cbind(all_data, "union") # performs gene cutoffs
#out = fastMNN(batch_sce, batch=colData(batch_sce)$batch)

batch_sce = read_python_cleaned_data("210222_cleaned_counts.csv", "210222_obs_batch_data.csv")

out = run_mnn(batch_sce)

# running and plotting tsne
combined_data_tsne = runTSNE(batch_sce)
batch_effect_plot = plotTSNE(combined_data_tsne, colour_by="batch")
ggsave("batch_effect_tsne_plot.png", batch_effect_plot)

tsne = runTSNE(out, dimred="corrected")
tsne_plot = plotTSNE(tsne, colour_by="batch")
ggsave("210125_batch_corrected_tsne_plot.png", tsne_plot)


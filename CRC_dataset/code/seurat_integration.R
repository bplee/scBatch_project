library(Seurat)
library(magrittr)
library(scran)
library(scater)
library(batchelor)


### APPLY FUNCTIONS ###
get_pat_id = function(string){
  rtn = strsplit(string, "_")[[1]][1]
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
  
  # Seurat expects genes to be the rows, cols to be the cells
  rtn = t(rtn)
  rtn = CreateSeuratObject(rtn)
  return(rtn)
}

get_clusters = function(file){
  # pat_id = get_pat_id(file)
  
  # first column contains the UMIs, so we use them as the row name
  rtn = read.csv(file, row.names=1)
  
  # CLUSTER is the first column, so we take it out:
  rtn = rtn$CLUSTER
  # nothing is being done with the cluster var. for now

  # Seurat expects genes to be the rows, cols to be the cells
  return(rtn)
}

### END FUNCTIONS ###

data_files = list.files()
pat_ids = sapply(data_files, get_pat_id)
# pat_ids is a vector

all_data = lapply(data_files, read_file)
# all data is a list, each element is a dataframe

# setting dataframe names as the patient ids
names(all_data) = pat_ids

# normalizing and reducing the feature size
for (i in 1:length(all_data)){
  all_data[[i]] = NormalizeData(all_data[[i]], verbose=F)
  all_data[[i]] = FindVariableFeatures(all_data[[i]], 
                                        selection.method = "vst", 
                                        nfeatures = 2000, 
                                        verbose = F)
}

# this step takes a while:
ref_anchors = FindIntegrationAnchors(object.list = all_data, dims = 1:30)

integrated_data <- IntegrateData(anchorset = ref_anchors, dims = 1:30)

# Now trying to visualize

library(ggplot2)
install.packages("cowplot")
install.packages("patchwork")
library(cowplot)
library(patchwork)

DefaultAssay(integrated_data) = "integrated"

integrated_data = ScaleData(integrated_data, verbose=F)
integrated_data = RunPCA(integrated_data, npcs=30, verbose=F)
integrated_data = RunUMAP(integrated_data, reduction="pca", dims=1:30)
p1 = Dimplot(integrated_data, reduction="umap", group.by="tech")
This directory contains code that allows us to run scnym.

The following is information regarding command line usage of scnym:

	Preprocessing:
		- Data input for scNym should be log(CPM + 1) normalized counts
		- Input data should be stored as a dense [Cells, Genes] CSV of 
		  normalized counts, in an AnnData h5ad obj, or a Loompy `loom` obj
		- An example of how to format your data is included below.

			adata = anndata.AnnData(
			  X = X, # [Cells, Genes] scipy.sparse.csr_matrix or numpy.ndarray
			  var = var, # [Genes, Features] pd.DataFrame with gene names as the index
			  obs = obs, # [Cells, Features] pd.DataFrame with cell barcodes as the index
			)

	Python API notes:
		- The API follows the scany functional style and has a single 
		  endpoint
		- This requires that the data be put into anndata.AnnData obj
		  from scanpy


An example of how to format your data is included below.

```python
adata = anndata.AnnData(
  X = X, # [Cells, Genes] scipy.sparse.csr_matrix or numpy.ndarray
  var = var, # [Genes, Features] pd.DataFrame with gene names as the index
  obs = obs, # [Cells, Features] pd.DataFrame with cell barcodes as the index
)

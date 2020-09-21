#!/bin/bash

#BSUB -q gpuqueue
#BSUB -J umap_graphs
#BSUB -n 10
#BSUB -gpu "num=1"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 3:00
#BSUB -o umap_plotting.stdout
#BSUB -e umap_plotting.stderr

python plot_umap_semi.py

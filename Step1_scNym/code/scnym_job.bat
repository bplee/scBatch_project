#!/bin/bash

#BSUB -q gpuqueue
#BSUB -J scnym_exp
#BSUB -n 10
#BSUB -gpu "num=1"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 5:00
#BSUB -o scnym.stdout
#BSUB -e scnym.stderr

conda env list
source activate test
conda env list

python run_scnym.py --test_patient 0

conda deactivate

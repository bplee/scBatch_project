#!/bin/bash

#BSUB -q gpuqueue
#BSUB -J starspace_exp
#BSUB -n 10
#BSUB -gpu "num=1"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 6:00
#BSUB -o starspace_test.stdout
#BSUB -e starspace_test.stderr


python run_starspace.py --test_patient 5

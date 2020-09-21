#!/bin/bash

#BSUB -q gpuqueue
#BSUB -J diva_exp
#BSUB -n 10
#BSUB -gpu "num=1"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 6:00
#BSUB -o diva_test.stdout
#BSUB -e diva_test.stderr


python rcc_exp_og_copy.py --test_patient 5 

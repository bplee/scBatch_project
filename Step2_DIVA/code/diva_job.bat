#!/bin/bash

#BSUB -q gpuqueue
#BSUB -J diva_exp
#BSUB -n 10
#BSUB -gpu "num=1"
#BSUB -R "rusage[mem=10GB]"
#BSUB -W 5:00
#BSUB -o diva_test.stdout
#BSUB -e diva_test.stderr


python new_rcc_exp.py  --test_patient 5 --y-dim 31


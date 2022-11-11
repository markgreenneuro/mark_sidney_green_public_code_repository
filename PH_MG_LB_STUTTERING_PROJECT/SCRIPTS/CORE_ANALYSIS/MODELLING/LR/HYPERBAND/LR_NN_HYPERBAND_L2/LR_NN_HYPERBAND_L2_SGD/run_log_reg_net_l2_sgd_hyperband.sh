#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=l2_sgd
#SBATCH --time=0-48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=/users/k1754828/LOG/CORE_ANALYSIS/MODELLING/LR/HYPERBAND/LR_NN_HYPERBAND_L2/LR_NN_HYPERBAND_L2_SGD/log_reg_net_hyperband_l2_sgd-%j.output
#SBATCH --error=/users/k1754828/LOG/CORE_ANALYSIS/MODELLING/LR/HYPERBAND/LR_NN_HYPERBAND_L2/LR_NN_HYPERBAND_L2_SGD/log_reg_net_hyperband_l2_sgd-%j.error

wd=/users/k1754828/SCRIPTS/CORE_ANALYSIS/MODELLING/LR/HYPERBAND/LR_NN_HYPERBAND_L2/LR_NN_HYPERBAND_L2_SGD


source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 main.py

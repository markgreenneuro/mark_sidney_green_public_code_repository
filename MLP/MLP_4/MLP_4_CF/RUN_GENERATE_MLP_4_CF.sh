#!/bin/bash -l
#SBATCH --exclusive
#SBATCH -c 6
#SBATCH --mem=59000
#SBATCH --job-name=generate_mlp_4_cf
#SBATCH --time=0-00:05:00
#SBATCH --partition=partition-1
#SBATCH --ntasks=1
#SBATCH --output=/home/markgreenneuroscience_gmail_com/LOG/ML/MLP/MLP_4/MLP_4_CF/generate_mlp_4_cf-%j.output
#SBATCH --error=/home/markgreenneuroscience_gmail_com/LOG/ML/MLP/MLP_4/MLP_4_CF/generate_mlp_4_cf-%j.error

wd=/home/markgreenneuroscience_gmail_com/ML/MLP/MLP_4/MLP_4_CF

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 GENERATE_MLP_4_CF.py

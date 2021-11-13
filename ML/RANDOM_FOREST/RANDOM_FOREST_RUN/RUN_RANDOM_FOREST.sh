#!/bin/bash -l
#SBATCH --exclusive
#SBATCH -c 6
#SBATCH --mem=59000
#SBATCH --job-name=random_forest
#SBATCH --partition=partition-1
#SBATCH --ntasks=1
#SBATCH --output=/home/markgreenneuroscience_gmail_com/LOG/ML/RANDOM_FOREST/RANDOM_FOREST_RUN/random_forest-%j.output
#SBATCH --error=/home/markgreenneuroscience_gmail_com/LOG/ML/RANDOM_FOREST/RANDOM_FOREST_RUN/random_forest-%j.error

wd=/home/markgreenneuroscience_gmail_com/ML/RANDOM_FOREST/RANDOM_FOREST_RUN

control_file_dir=/home/markgreenneuroscience_gmail_com/ML/RANDOM_FOREST/RANDOM_FOREST_CF
random_forest_control_file=${control_file_dir}/RANDOM_FOREST_CF.txt
splits=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${random_forest_control_file} | awk '{print $1}')

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 RANDOM_FOREST.py "$splits"

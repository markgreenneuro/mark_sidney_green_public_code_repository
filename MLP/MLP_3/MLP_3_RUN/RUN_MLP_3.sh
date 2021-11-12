#!/bin/bash -l
#SBATCH --exclusive
#SBATCH -c 6
#SBATCH --cpus-per-task=1
#SBATCH --mem=59000
#SBATCH --job-name=mlp_3
#SBATCH --partition=partition-1
#SBATCH --ntasks=1
#SBATCH --output=/home/markgreenneuroscience_gmail_com/LOG/ML/MLP/MLP_3/MLP_3_RUN/mlp_3-%j.output
#SBATCH --error=/home/markgreenneuroscience_gmail_com/LOG/ML/MLP/MLP_3/MLP_3_RUN/mlp_3-%j.error

wd=/home/markgreenneuroscience_gmail_com/ML/MLP/MLP_3/MLP_3_RUN

control_file_dir=/home/markgreenneuroscience_gmail_com/ML/MLP/MLP_3/MLP_3_CF
mlp_3_control_file=${control_file_dir}/MLP_3_CF.txt
ni_neurons=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${mlp_3_control_file} | awk '{print $1}')
nj_neurons=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${mlp_3_control_file} | awk '{print $2}')
nk_neurons=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${mlp_3_control_file} | awk '{print $3}')
epochs=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${mlp_3_control_file} | awk '{print $4}')
batch_size=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${mlp_3_control_file} | awk '{print $5}')
optimizer=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${mlp_3_control_file} | awk '{print $6}')
activation=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${mlp_3_control_file} | awk '{print $7}')

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 MLP_3.py "$ni_neurons" "$nj_neurons" "$nk_neurons" "$epochs" "$batch_size" "$optimizer" "$activation"

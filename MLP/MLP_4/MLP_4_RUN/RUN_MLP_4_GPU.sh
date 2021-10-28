#!/bin/bash -l
#SBATCH --exclusive
#SBATCH -c 6
#SBATCH --mem=59000
#SBATCH --job-name=mlp_4_gpu
#SBATCH --partition=partition-1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/markgreenneuroscience_gmail_com/LOG/ML/MLP/MLP_4/MLP_4_GPU_RUN/mlp_4_gpu-%j.output
#SBATCH --error=/home/markgreenneuroscience_gmail_com/LOG/ML/MLP/MLP_4/MLP_4_GPU_RUN/mlp_4_gpu-%j.error

wd=/home/markgreenneuroscience_gmail_com/MODELLING/MLP/MLP_4/MLP_4_RUN

control_file_dir=/home/markgreenneuroscience_gmail_com/MODELLING/MLP/MLP_4/MLP_4_CF
mlp_4_control_file=${control_file_dir}/MLP_4_CF.txt
ni_neurons=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${mlp_4_control_file} | awk '{print $1}')
nj_neurons=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${mlp_4_control_file} | awk '{print $2}')
nk_neurons=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${mlp_4_control_file} | awk '{print $3}')
nl_neurons=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${mlp_4_control_file} | awk '{print $4}')
epochs=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${mlp_4_control_file} | awk '{print $5}')
batch_size=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${mlp_4_control_file} | awk '{print $6}')
optimizer=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${mlp_4_control_file} | awk '{print $7}')
activation=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${mlp_4_control_file} | awk '{print $8}')

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 MLP_4.py "$ni_neurons" "$nj_neurons" "$nk_neurons" "$nl_neurons" "$epochs" "$batch_size" "$optimizer" "$activation"

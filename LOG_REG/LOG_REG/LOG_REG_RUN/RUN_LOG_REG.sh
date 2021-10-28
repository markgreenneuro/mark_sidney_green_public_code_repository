#!/bin/bash -l
#SBATCH -c 6
#SBATCH --mem=59000
#SBATCH --job-name=log_reg
#SBATCH --partition=partition-1
#SBATCH --nodes=1
#SBATCH --output=/home/markgreenneuroscience_gmail_com/LOG/ML/LOG_REG/LOG_REG_RUN/log_reg-%j.output
#SBATCH --error=/home/markgreenneuroscience_gmail_com/LOG/ML/LOG_REG/LOG_REG_RUN/log_reg-%j.error

wd=/home/markgreenneuroscience_gmail_com/ML/LOG_REG/LOG_REG_RUN

control_file_dir=/home/markgreenneuroscience_gmail_com/ML/LOG_REG/LOG_REG/LOG_REG_ONE_CF
log_reg_control_file=${control_file_dir}/LOG_REG_ONE_CF.txt
penalty_term=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${log_reg_control_file} | awk '{print $1}')
c_term=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${log_reg_control_file} | awk '{print $2}')
solver_term=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${log_reg_control_file} | awk '{print $3}')
multi_class_term=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${log_reg_control_file} | awk '{print $4}')
elastic_net_term=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${log_reg_control_file} | awk '{print $5}')

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate myenv

cd $wd || exit
python3 LOG_REG.py "$penalty_term" "$c_term" "$solver_term" "$multi_class_term" "$elastic_net_term"

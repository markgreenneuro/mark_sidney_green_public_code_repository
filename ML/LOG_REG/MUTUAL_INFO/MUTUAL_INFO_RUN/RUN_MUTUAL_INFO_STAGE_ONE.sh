#!/bin/bash -l
#SBATCH -c 6
#SBATCH --mem=59000
#SBATCH --job-name=mutual_info_stage_one
#SBATCH --time=0-24:00:00
#SBATCH --partition=partition-1
#SBATCH --nodes=1
#SBATCH --output=/home/markgreenneuroscience_gmail_com/LOG/ML/LOG_REG/MUTUAL_INFO/MUTUAL_INFO_STAGE_ONE_RUN/mutual_info_stage_one-%j.output
#SBATCH --error=/home/markgreenneuroscience_gmail_com/LOG/ML/LOG_REG/MUTUAL_INFO/MUTUAL_INFO_STAGE_ONE_RUN/mutual_info_stage_one-%j.error

wd=/home/markgreenneuroscience_gmail_com/ML/LOG_REG/MUTUAL_INFO/MUTUAL_INFO_RUN

control_file_dir=/home/markgreenneuroscience_gmail_com/ML/LOG_REG/MUTUAL_INFO/MUTUAL_INFO_STAGE_ONE_CF
mutual_info_stage_one_file=${control_file_dir}/MUTUAL_INFO_STAGE_ONE_CF.txt
feat_num=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${mutual_info_stage_one_file} | awk '{print $1}')

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 MUTUAL_INFO_EXTRACTION_BIC.py "$feat_num"


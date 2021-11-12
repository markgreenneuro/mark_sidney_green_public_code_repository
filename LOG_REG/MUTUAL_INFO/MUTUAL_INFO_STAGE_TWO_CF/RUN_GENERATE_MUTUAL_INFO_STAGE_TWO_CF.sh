#!/bin/bash -l
#SBATCH --exclusive
#SBATCH -c 6
#SBATCH --mem=59000
#SBATCH --job-name=mutual_info_stage_two_cf
#SBATCH --time=0-00:05:00
#SBATCH --partition=partition-1
#SBATCH --ntasks=1
#SBATCH --output=/home/markgreenneuroscience_gmail_com/LOG/ML/LOG_REG/MUTUAL_INFO/MUTUAL_INFO_STAGE_TWO_CF/mutual_info_stage_two_cf-%j.output
#SBATCH --error=/home/markgreenneuroscience_gmail_com/LOG/ML/LOG_REG/MUTUAL_INFO/MUTUAL_INFO_STAGE_TWO_CF/mutual_info_stage_two_cf-%j.error

wd=/home/markgreenneuroscience_gmail_com/ML/LOG_REG/MUTUAL_INFO/MUTUAL_INFO_STAGE_TWO_CF

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 GENERATE_MUTUAL_INFO_STAGE_TWO_CF.py

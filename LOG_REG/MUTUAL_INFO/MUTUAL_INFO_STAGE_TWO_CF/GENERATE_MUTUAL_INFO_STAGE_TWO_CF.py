#!/usr/bin/env python
# coding: utf-8
import numpy as np
from os import listdir, chdir
from os.path import isfile, join
import pandas as pd
from pathlib import Path
import sys


def set_stats_root():
    stats_dir = Path('/home/markgreenneuroscience_gmail_com/RESULTS/LOG_REG/MUTUAL_INFO_STATS')
    return stats_dir


def set_control_file_root():
    control_file_dir = Path('/home/markgreenneuroscience_gmail_com/ML/LOG_REG/MUTUAL_INFO/MUTUAL_INFO_STAGE_TWO_CF')
    return control_file_dir


def generate_full_gof_file(stats_dir):
    files = pd.DataFrame([f for f in listdir(stats_dir) if isfile(join(stats_dir, f))], columns=['FILES'])
    gof_files = files[files['FILES'].str.contains('gof')].reset_index(drop=True)
    chdir(stats_dir)
    full_gof_file = pd.DataFrame([])
    for i in range(gof_files.shape[0]):
        single_gof_file = pd.read_csv(gof_files.iloc[i][0])
        full_gof_file = pd.concat([full_gof_file, single_gof_file], axis=0)
    return full_gof_file


def find_winning_num_feat(full_gof_file):
    full_gof_file = full_gof_file.sort_values(by=['FEAT_NUM']).reset_index(drop=True)
    full_gof_file.columns = ['FEAT_NUM', 'MI_SUM', 'AIC', 'BIC']
    winning_row = pd.DataFrame(full_gof_file.iloc[full_gof_file['BIC'].argmin()]).T
    winning_num_feat = int(winning_row['FEAT_NUM'][0])
    return winning_num_feat


def write_control_file(winning_feat_num, control_file_name, control_file_dir):
    winning_feat_num_array = np.repeat([winning_feat_num], 8)
    int_to_add_array = np.array([-1, -2, -3, -4, 1, 2, 3, 4])
    new_feat_num_to_search = winning_feat_num_array + int_to_add_array
    new_feat_num_to_search = int(np.sort(new_feat_num_to_search)[0])
    chdir(control_file_dir)
    np.savetxt(control_file_name, new_feat_num_to_search, fmt='%s', delimiter=',')


def run_all_generate_mutual_info_stage_two_cf() -> None:
    control_file_name = 'MUTUAL_INFO_STAGE_TWO_CF.txt'
    stats_dir = set_stats_root()
    control_file_dir = set_control_file_root()
    full_gof_file = generate_full_gof_file(stats_dir)
    winning_num_feat = find_winning_num_feat(full_gof_file)
    write_control_file(winning_num_feat, control_file_name, control_file_dir)
    sys.exit('MUTUAL INFO STAGE TWO CONTROL FILE CREATED SUCCESSFULLY')


# RUN ALL
run_all_generate_mutual_info_stage_two_cf()

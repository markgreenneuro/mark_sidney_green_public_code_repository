#!/usr/bin/env python
# coding: utf-8
from datetime import datetime
from os import listdir, chdir
from os.path import isfile, join
import pandas as pd
from pathlib import Path
import sys


def set_stats_root():
    stats_dir = Path('/home/markgreenneuroscience_gmail_com/RESULTS/LOG_REG/MUTUAL_INFO_STATS')
    return stats_dir


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
    winning_feat_num = int(winning_row['FEAT_NUM'][0])
    return winning_feat_num


def set_data_root():
    data_dir = Path('/home/markgreenneuroscience_gmail_com/DATA/MASTER_FEATURE')
    return data_dir


def get_uncut_matrix(matrix_dir):
    uncut_matrix_name = 'speech_features_kf_ts.csv'
    chdir(matrix_dir)
    uncut_matrix = pd.read_csv(uncut_matrix_name)
    return uncut_matrix


def perform_cut_of_matrix(winning_feat_num, uncut_matrix, stats_dir):
    name_of_save_file: str = 'FEATURE_IMPORTANCE_' + str(winning_feat_num) + '.csv'
    chdir(stats_dir)
    winning_features = pd.read_csv(name_of_save_file)
    cut_matrix = uncut_matrix[winning_features['FEATS']]
    return cut_matrix


def save_cut_matrix(cut_matrix, matrix_dir):
    cut_matrix_name = 'speech_features_kf_ts_mi_bi_cut.csv'
    chdir(matrix_dir)
    cut_matrix.to_csv(cut_matrix_name)


def run_all_find_winning_model_and_cut_matrix() -> None:
    start_time = datetime.now()
    stats_dir = set_stats_root()
    full_gof_file = generate_full_gof_file(stats_dir)
    winning_feat_num = find_winning_num_feat(full_gof_file)
    data_dir = set_data_root()
    uncut_matrix = get_uncut_matrix(data_dir)
    cut_matrix = perform_cut_of_matrix(winning_feat_num, uncut_matrix, stats_dir)
    save_cut_matrix(cut_matrix, data_dir)
    time_delta = datetime.now() - start_time
    exit_message = 'CUT MATRIX IN: ' + str(time_delta)
    sys.exit(exit_message)


# RUN ALL
run_all_find_winning_model_and_cut_matrix()

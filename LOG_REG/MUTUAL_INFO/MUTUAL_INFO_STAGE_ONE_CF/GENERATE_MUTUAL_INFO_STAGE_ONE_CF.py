#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import pandas as pd
import sys


def set_data_root():
    data_dir = '/home/markgreenneuroscience_gmail_com/DATA/MASTER_FEATURE'
    return data_dir


def set_output_root():
    output_dir = '/home/markgreenneuroscience_gmail_com/ML/LOG_REG/MUTUAL_INFO/MUTUAL_INFO_STAGE_ONE_CF'
    return output_dir


def get_file(data_dir):
    os.chdir(data_dir)
    mf = pd.read_csv('speech_features_kf_ts.csv')
    return mf


def write_control_file(mf, output_dir):
    mf = mf.drop(['stutter'], axis=1)
    [_, c] = mf.shape
    os.chdir(output_dir)
    num_feat_list = pd.DataFrame(np.arange(10, c, 10).tolist())
    c_df = pd.DataFrame([c])
    num_feat_list = pd.concat([num_feat_list, c_df], axis=0).reset_index(drop=True)
    num_feat_list.columns = ['NUM_FEAT']
    num_feat_list.to_csv('MUTUAL_INFO_STAGE_ONE_CF.txt', index=False, header=False)


def run_all_generate_mutual_info_stage_one_cf() -> None:
    data_dir = set_data_root()
    output_dir = set_output_root()
    mf = get_file(data_dir)
    write_control_file(mf, output_dir)
    sys.exit('MUTUAL INFO STAGE ONE CONTROL FILE CREATED SUCCESSFULLY')


# RUN_ALL
run_all_generate_mutual_info_stage_one_cf()

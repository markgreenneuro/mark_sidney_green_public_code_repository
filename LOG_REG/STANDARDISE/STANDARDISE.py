#!/usr/bin/env python
# coding: utf-8#
from datetime import datetime
import numpy as np
from os import chdir
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import sys


def set_data_root():
    data_dir = Path('/home/markgreenneuroscience_gmail_com/DATA/MASTER_FEATURE')
    return data_dir


def get_file(data_dir):
    chdir(data_dir)
    mf = pd.read_csv('speech_features_kf_ts_mi_bic_cut.csv')
    return mf


def split(mf):
    stutter = pd.DataFrame(mf.stutter)
    x = pd.DataFrame(mf.loc[:, mf.columns != 'stutter'])
    return [x, stutter]


def standardise(x):
    x = pd.DataFrame(x)
    master_idx = x.master_idx
    sess_idx = x.sess_idx
    speaker_id = x.speaker_id
    sess_id = x.sess_id
    x = x.loc[:, x.columns != 'master_idx']
    x = x.loc[:, x.columns != 'sess_idx']
    x = x.loc[:, x.columns != 'speaker_id']
    x = x.loc[:, x.columns != 'sess_id']
    scaler = StandardScaler()
    scaler.fit(x)
    x = pd.DataFrame(scaler.transform(x))
    x = pd.concat([master_idx, sess_idx, speaker_id, sess_id, x], axis=1)
    return x


def merge(x, stutter):
    x_y = pd.concat([x, stutter], axis=1)
    return x_y


def save_standardised_matrix(x_y, data_dir):
    standardised_matrix_name = 'speech_features_kf_ts_mi_bic_cut_standardised.csv'
    chdir(data_dir)
    x_y.to_csv(standardised_matrix_name)


def run_all_standardisation_script() -> None:
    start_time = datetime.now()
    # SET SEED HERE
    np.random.seed(1234)
    data_dir = set_data_root()
    mf = get_file(data_dir)
    [x, stutter] = split(mf)
    x = standardise(x)
    x_y = merge(x, stutter)
    save_standardised_matrix(x_y, data_dir)
    time_delta = datetime.now() - start_time
    exit_message = 'STANDARDISE RAN SUCCESSFULLY IN: ' + str(time_delta)
    sys.exit(exit_message)


# RUN ALL
run_all_standardisation_script()

#!/usr/bin/env python
# coding: utf-8
from datetime import datetime
import numpy as np
from os import chdir
import pandas as pd
from pathlib import Path
from sklearn.utils import resample
import sys


def set_data_root():
    data_dir = Path('/home/markgreenneuroscience_gmail_com/DATA/MASTER_FEATURE')
    return data_dir


def get_file(data_dir):
    chdir(data_dir)
    x_y = pd.read_csv('speech_features_kf_ts_mi_bic_cut_standardised.csv')
    return x_y


def split_train_test_cases(x_y, percentage_in_test_set):
    x_y_test_indicies = np.array(resample(np.unique(x_y.speaker_id),
                                          n_samples=int(len(np.unique(x_y.speaker_id)) * percentage_in_test_set // 100),
                                          random_state=0))
    x_y_train_indicies = np.array(list(set(np.unique(x_y.speaker_id)) - set(x_y_test_indicies)))
    if len(x_y_test_indicies) == 0:
        sys.exit("TOO FEW UNIQUE SPEAKERS")
    else:
        pass
    x_y_train = x_y.loc[x_y.speaker_id.isin(x_y_train_indicies)].reset_index(drop=True)
    x_y_test = x_y.loc[x_y.speaker_id.isin(x_y_test_indicies)].reset_index(drop=True)
    return [x_y_train, x_y_test]


def save_split_train_test_cases(data_dir, x_y_train, x_y_test):
    chdir(data_dir)
    name_x_y_train = 'speech_features_kf_ts_mi_bic_cut_standardised_train.csv'
    name_x_y_test = 'speech_features_kf_ts_mi_bic_cut_standardised_test.csv'
    x_y_train.to_csv(name_x_y_train)
    x_y_test.to_csv(name_x_y_test)


def run_all_split_train_test(percentage_in_test_set) -> None:
    start_time = datetime.now()
    np.random.seed(1234)
    data_dir = set_data_root()
    x_y = get_file(data_dir)
    [x_y_train, x_y_test] = split_train_test_cases(x_y, percentage_in_test_set)
    save_split_train_test_cases(data_dir, x_y_train, x_y_test)
    time_delta = datetime.now() - start_time
    exit_message = 'SPLIT RAN SUCCESSFULLY: ' + str(time_delta)
    sys.exit(exit_message)


# RUN ALL
run_all_split_train_test(20)

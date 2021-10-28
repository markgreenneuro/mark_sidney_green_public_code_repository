#!/usr/bin/env python
# coding: utf-8
from datetime import datetime
import numpy as np
from os import chdir, makedirs
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import sys


def get_feat_num():
    feat_num = int(sys.argv[1])
    return feat_num


def set_data_root():
    data_dir = Path('/home/markgreenneuroscience_gmail_com/DATA/MASTER_FEATURE')
    return data_dir


def set_log_reg_stats_root():
    log_reg_dir = Path('/home/markgreenneuroscience_gmail_com/RESULTS/LOG_REG/')
    return log_reg_dir


def get_file(data_dir):
    chdir(data_dir)
    mf = pd.read_csv('speech_features_kf_ts.csv')
    return mf


def split(mf):
    stutter = pd.DataFrame(mf.stutter)
    feats = list(set(list(mf.columns)) - set(list(stutter.columns)))
    x = mf[feats]
    return [x, stutter, feats]


def select_features(x, stutter, feat_num):
    fs = SelectKBest(score_func=mutual_info_classif, k=feat_num)
    fs.fit(x, np.array(stutter).ravel())
    return fs


def calculate_mutual_information(fs, feats) -> pd.DataFrame:
    new_feats = pd.DataFrame(feats)[fs.get_support()].reset_index(drop=True)
    new_feats.columns = ['FEATS']
    mi = pd.DataFrame(fs.scores_)[fs.get_support()].reset_index(drop=True)
    mi.columns = ['MUTUAL_INFO']
    new_feats_mi = pd.concat([new_feats, mi], axis=1)
    [r, _] = new_feats_mi.shape
    num_feats = pd.DataFrame([np.repeat(str(r), r)]).T
    num_feats.columns = ['NUM_FEATS']
    new_feats_mi = pd.concat([num_feats, new_feats_mi], axis=1)
    return new_feats_mi


def get_make_stats_root(log_reg_dir):
    stats_dir = Path.home().joinpath(log_reg_dir, str('MUTUAL_INFO_STATS'))
    if stats_dir.exists():
        pass
    else:
        makedirs(stats_dir)
    stats_dir = str(stats_dir)
    return stats_dir


def output_feature_importances(new_feats_mi, stats_dir):
    num_feats = (np.array(new_feats_mi.NUM_FEATS).ravel())[0]
    name_of_save_file: str = 'FEATURE_IMPORTANCE_' + str(num_feats) + '.csv'
    chdir(stats_dir)
    new_feats_mi.to_csv(name_of_save_file, index=False)


def get_feature_importances(feat_num):
    """
    Get feature importances for feat_num
    :param feat_num:
    """
    data_dir = set_data_root()
    log_reg_dir = set_log_reg_stats_root()
    stats_dir = get_make_stats_root(log_reg_dir)
    mf = get_file(data_dir)
    [x, stutter, feats] = split(mf)
    fs = select_features(x, stutter, feat_num)
    new_feats_mi = calculate_mutual_information(fs, feats)
    output_feature_importances(new_feats_mi, stats_dir)
    return new_feats_mi


def get_params(new_feats_mi):
    num_params = new_feats_mi.shape[0]
    n = new_feats_mi.shape[1]
    return [num_params, n]


def find_mi_sum(new_feats_mi):
    mi_sum = float(np.sum(new_feats_mi['MUTUAL_INFO']))
    return mi_sum


def calculate_aic(n: int, mi_sum: float, num_params):
    if mi_sum != 0:
        aic = n * np.log(mi_sum) + 2 * num_params
    else:
        aic = 2 * num_params
    return aic


def calculate_bic(n: int, mi_sum: float, num_params: int) -> float:
    if mi_sum == 0:
        bic = num_params * np.log(n)
    else:
        bic = n * np.log(mi_sum) + num_params * np.log(n)
    return bic


def get_gof_statistics(new_feats_mi):
    [num_params, n] = get_params(new_feats_mi)
    mi_sum = find_mi_sum(new_feats_mi)
    aic = calculate_aic(n, mi_sum, num_params)
    bic = calculate_bic(n, mi_sum, num_params)
    return [mi_sum, aic, bic]


def export_gof_df(feat_num, mi_sum, aic, bic):
    log_reg_dir = set_log_reg_stats_root()
    stats_dir = get_make_stats_root(log_reg_dir)

    feat_num_df = pd.DataFrame([feat_num], columns=['FEAT_NUM'])
    mi_sum_df = pd.DataFrame([mi_sum], columns=['MI_SUM'])
    aic_df = pd.DataFrame([aic], columns=['AIC'])
    bic_df = pd.DataFrame([bic], columns=['BIC'])

    gof_df = pd.concat([feat_num_df, mi_sum_df, aic_df, bic_df], axis=1)
    gof_df_name = 'gof_' + str(feat_num) + '.csv'
    chdir(stats_dir)
    gof_df.to_csv(gof_df_name, index=False)


def run_all_mutual_information_extraction_bic() -> None:
    start_time = datetime.now()
    feat_num = get_feat_num()
    new_feats_mi = get_feature_importances(feat_num)
    [mi_sum, aic, bic] = get_gof_statistics(new_feats_mi)
    export_gof_df(feat_num, mi_sum, aic, bic)
    time_delta = datetime.now() - start_time
    exit_message = 'MUTUAL INFORMATION EXTRACTION RAN SUCCESSFULLY IN: ' + str(time_delta)
    sys.exit(exit_message)


# RUN_ALL
run_all_mutual_information_extraction_bic()

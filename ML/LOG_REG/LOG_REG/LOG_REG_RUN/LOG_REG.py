#!/usr/bin/env python
# coding: utf-8
from datetime import datetime
from joblib import dump
from os import chdir, makedirs
from pathlib import Path
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, mean_absolute_error
from sklearn.preprocessing import label_binarize

from LOG_REG_ONE_CONF_MAT_CLASS_REPORT_FORMAT import *
from LOG_REG_ONE_CONF_MAT_CLASS_REPORT_OUTPUT import *
from LOG_REG_ONE_SUMMARY_STATS_OUTPUT import *


def get_params():
    """
    Belts and braces with explicit cast and object specification.
    :return: [feat_num, penalty_term, dual_term, c_term, multi_class_term, solver_term]
    """
    penalty_term: str = str(sys.argv[1])
    c_term: float = float(sys.argv[2])
    solver_term: str = str(sys.argv[4])
    multi_class_term: str = str(sys.argv[3])
    elastic_net_term: float = float(sys.argv[5])

    return [penalty_term, c_term, solver_term, multi_class_term, elastic_net_term]


def set_data_root():
    data_dir = Path('/home/markgreenneuroscience_gmail_com/DATA/MASTER_FEATURE')
    return data_dir


def get_files(data_dir):
    chdir(data_dir)
    x_y_train = pd.read_csv('speech_features_kf_ts_mi_bic_cut_standardised_train.csv')
    x_y_test = pd.read_csv('speech_features_kf_ts_mi_bic_cut_standardised_test.csv')
    return [x_y_train, x_y_test]


def set_log_reg_root():
    log_reg_dir = Path('/home/markgreenneuroscience_gmail_com/RESULTS/LOG_REG/')
    if log_reg_dir.exists():
        pass
    else:
        makedirs(log_reg_dir)
    log_reg_dir = str(log_reg_dir)
    return log_reg_dir


def get_make_stats_root(log_reg_dir):
    stats_dir = Path.home().joinpath(log_reg_dir, str('LOG_REG_STATS'))
    if stats_dir.exists():
        pass
    else:
        makedirs(stats_dir)
    stats_dir = str(stats_dir)
    return stats_dir


def logistic_regression(x_y_train, penalty_term, solver_term, c_term, multi_class_term, elastic_net_term):
    stutter_train = x_y_train.stutter
    x_train = x_y_train.loc[:, x_y_train.columns != 'stutter']

    if penalty_term == 'elasticnet':
        log_reg = LogisticRegression(penalty=penalty_term, C=c_term,
                                     max_iter=100000, solver=solver_term, multi_class=multi_class_term,
                                     l1_ratio=elastic_net_term, n_jobs=-1)
    else:
        log_reg = LogisticRegression(penalty=penalty_term, C=c_term,
                                     max_iter=100000, solver=solver_term, multi_class=multi_class_term, n_jobs=-1)
    log_reg.fit(x_train, np.array(stutter_train))
    return log_reg


def classification_statistics(log_reg, x_y_test):
    stutter_test = x_y_test.stutter
    x_test = x_y_test.loc[:, x_y_test.columns != 'stutter']
    stutter_pred = log_reg.predict(x_test)
    reg_score = log_reg.score(x_test, stutter_test)
    class_report = classification_report(stutter_test, stutter_pred, labels=np.unique(stutter_pred), output_dict=True)
    conf_mat = confusion_matrix(stutter_test, stutter_pred, labels=np.unique(stutter_pred))
    return [reg_score, class_report, stutter_pred, conf_mat]


def get_stutter_test(x_y_test):
    stutter_test = x_y_test.stutter
    return stutter_test


def compute_roc_and_auc(stutter_test, stutter_pred, x_y_train):
    fpr = dict()
    tpr = dict()
    stutter_test = label_binarize(stutter_test, classes=np.unique(x_y_train.stutter))
    stutter_pred = label_binarize(stutter_pred, classes=np.unique(x_y_train.stutter))
    roc_auc = dict()
    for i in range(0, len(np.unique(x_y_train.stutter))):
        fpr[i], tpr[i], _ = roc_curve(stutter_test[:, i], stutter_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # COMPUTE MICRO AVERAGE ROC CURVE
    fpr["micro"], tpr["micro"], _ = roc_curve(stutter_test.ravel(), stutter_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    roc_auc = roc_auc["micro"]
    return roc_auc


def get_num_params(log_reg):
    num_params = len(log_reg.get_params(deep=True)) + 1
    return num_params


def calculate_aic(n, mse: float, num_params):
    if mse != 0:
        aic = n * np.log(mse) + 2 * num_params
    else:
        aic = 2 * num_params
    return aic


def calculate_bic(n: float, mse: float, num_params: int) -> float:
    if mse == 0:
        bic = num_params * np.log(n)
    else:
        bic = n * np.log(mse) + num_params * np.log(n)
    return bic


def get_mse_bic_aic(stutter_test, stutter_pred, num_params):
    mse = mean_absolute_error(stutter_test, stutter_pred)
    aic = calculate_aic(len(stutter_test), mse, num_params)
    bic = calculate_bic(len(stutter_test), mse, num_params)
    return [mse, aic, bic]


def get_make_modelsave_root(log_reg_dir):
    modelsave_dir = Path.home().joinpath(log_reg_dir, str('LOG_REG_MODELS'))
    if modelsave_dir.exists():
        pass
    else:
        makedirs(modelsave_dir)
    modelsave_dir = str(modelsave_dir)
    return modelsave_dir


def write_modelsave(log_reg_dir, log_reg, penalty_term, c_term, multi_class_term, solver_term, elastic_net_term):
    modelsave_dir = get_make_modelsave_root(log_reg_dir)
    chdir(modelsave_dir)
    name = str(penalty_term) + '_' + str(c_term) + '_' + str(multi_class_term) + '_' + str(solver_term) + '_' + str(
        elastic_net_term) + '.model'
    dump(log_reg, name)


def run_all_logistic_regression() -> None:
    start_time = datetime.now()
    [penalty_term, c_term, multi_class_term, solver_term, elastic_net_term] = get_params()
    data_dir = set_data_root()
    [x_y_train, x_y_test] = get_files(data_dir)
    log_reg_dir = set_log_reg_root()
    stats_dir = get_make_stats_root(log_reg_dir)
    log_reg = logistic_regression(x_y_train, penalty_term, solver_term, c_term, multi_class_term, elastic_net_term)
    [reg_score, class_report, stutter_pred, conf_mat] = classification_statistics(log_reg, x_y_test)
    stutter_test = get_stutter_test(x_y_test)
    roc_auc = compute_roc_and_auc(stutter_test, stutter_pred, x_y_train)
    num_params = get_num_params(log_reg)
    [mse, aic, bic] = get_mse_bic_aic(stutter_test, stutter_pred, num_params)
    [conf_mat, class_report] = class_report_conf_mat(conf_mat, class_report, penalty_term, c_term, multi_class_term,
                                                     elastic_net_term, solver_term, stutter_test)
    output_conf_mat_class_report(penalty_term, c_term, multi_class_term, solver_term, elastic_net_term,
                                 conf_mat, class_report, stats_dir)
    output_summary_stats(reg_score, roc_auc, mse, bic, aic, penalty_term, c_term, multi_class_term,
                         elastic_net_term, solver_term, stats_dir)
    write_modelsave(log_reg_dir, log_reg, penalty_term, c_term, multi_class_term, solver_term, elastic_net_term)
    time_delta = datetime.now() - start_time
    exit_message = 'LOGISTIC REGRESSION RAN SUCCESSFULLY IN: ' + str(time_delta)
    sys.exit(exit_message)


# RUN_ALL
run_all_logistic_regression()

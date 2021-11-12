# !/usr/bin/env python
# coding: utf-8
from itertools import product
import numpy as np
import os
import pandas as pd
import sys


# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

def set_output_root():
    output_dir = '/home/markgreenneuroscience_gmail_com/ML/PRELIMINARY_LOG_REG/PRELIMINARY_LOG_REG_CF'
    return output_dir


def initialise_parameter_mat():
    """
    :return: return raw parameter matrix, pre-cutting forbidden combinations
    """
    penalty = ['l1', 'l2', 'none']
    c = [x / 10.0 for x in range(1, 11)]
    solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    multiclass = ['ovr', 'multinomial']
    elastic_net_mixing_parameter = [np.nan]
    parameter_mat = pd.DataFrame(product(penalty, c, solver, multiclass, elastic_net_mixing_parameter),
                                 columns=['penalty', 'c', 'solver', 'multiclass',
                                          'elastic_net_mixing_parameter'])
    return parameter_mat


def initialise_elastic_net_parameter_mat():
    """
    Elastic_net is supported only by the saga solver
    :return: elastic_net_parameter_mat
    :return: return raw elastic net parameter matrix, pre-cutting forbidden combinations
    """
    penalty = ['elasticnet']
    c = [x / 10.0 for x in range(0, 11)]
    solver = ['saga']
    multiclass = ['ovr', 'multinomial']
    elastic_net_mixing_parameter = [x / 10.0 for x in range(1, 9)]
    elastic_net_parameter_mat = pd.DataFrame(product(penalty, c, solver, multiclass, elastic_net_mixing_parameter),
                                             columns=['penalty', 'c', 'solver', 'multiclass',
                                                      'elastic_net_mixing_parameter'])
    return elastic_net_parameter_mat


def join_parameter_mats(parameter_mat, elastic_net_parameter_mat):
    parameter_mat = pd.concat([parameter_mat, elastic_net_parameter_mat], axis=0).reset_index(drop=True)
    return parameter_mat


def l2_penalty_specific_solvers(parameter_mat):
    """
    The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties.
    :param parameter_mat: modified parameter_mat
    :return: parameter_mat
    """
    parameter_mat = pd.concat([parameter_mat[(parameter_mat['penalty'].isin(['l1', 'elasticnet'])) & (
        parameter_mat['solver'].isin(['newton-cg', 'lbfgs', 'sag']))].reset_index(drop=True),
                               parameter_mat]).drop_duplicates(keep=False).reset_index(drop=True)
    return parameter_mat


def liblinear_ovr(parameter_mat):
    """
    The 'liblinear' is limited to one-versus-rest schemes.
    :param parameter_mat: modified parameter_mat
    :return: parameter_mat
    """
    parameter_mat = pd.concat([parameter_mat[(parameter_mat['solver'].isin(['liblinear'])) & (
        parameter_mat['multiclass'].isin(['multinomial']))].reset_index(drop=True), parameter_mat]).drop_duplicates(
        keep=False).reset_index(drop=True)
    return parameter_mat


def multinomial(parameter_mat):
    """
    For multiclass problems only 'newton-cg', 'sag', 'saga' and 'lbfgs' handle multinomial loss.
    :param parameter_mat: modified parameter_mat
    :return: parameter_mat
    """
    parameter_mat = pd.concat([parameter_mat[(parameter_mat['solver'].isin(['newton-cg', 'sag', 'saga', 'lbfgs'])) & (
        parameter_mat['multiclass'].isin(['ovo']))].reset_index(drop=True), parameter_mat]).drop_duplicates(
        keep=False).reset_index(drop=True)
    return parameter_mat


def remove_duplicates_helper_function(parameter_mat):
    """
    Remove parameter_mat duplicated columns
    :param : parameter_mat: modified parameter_mat
    :return: parameter_mat
    """
    parameter_mat = pd.concat([parameter_mat[(parameter_mat['penalty'].isin(['elasticnet'])) & (
        parameter_mat['elastic_net_mixing_parameter'].isin(['NaN']))].reset_index(drop=True),
                               parameter_mat]).drop_duplicates(
        keep=False).reset_index(drop=True)

    parameter_mat = parameter_mat.loc[:, ~parameter_mat.columns.duplicated()]
    return parameter_mat


def create_parameter_mat():
    """
    Run initial parameter creation
    :return: parameter_mat
    """
    parameter_mat = initialise_parameter_mat()
    elastic_net_parameter_mat = initialise_elastic_net_parameter_mat()
    parameter_mat = join_parameter_mats(parameter_mat, elastic_net_parameter_mat)
    parameter_mat = l2_penalty_specific_solvers(parameter_mat)
    parameter_mat = liblinear_ovr(parameter_mat)
    parameter_mat = multinomial(parameter_mat)
    parameter_mat = remove_duplicates_helper_function(parameter_mat)
    return parameter_mat


def output_formatted_parameter_mat(parameter_mat):
    """
    Output parameter_mat
    :param parameter_mat: modified parameter_mat convolved with num_feat_list
    """
    output_dir = set_output_root()
    os.chdir(output_dir)
    parameter_mat = np.array(parameter_mat)
    np.savetxt('PRELIMINARY_LOG_REG_CF.txt', parameter_mat, fmt='%s', delimiter='\t')


def run_all_generate_preliminary_logistic_regression_cf() -> None:
    """
    Assemble and write parameter control file
    """
    output_dir = set_output_root()
    os.chdir(output_dir)
    parameter_mat = create_parameter_mat()
    output_formatted_parameter_mat(parameter_mat)
    sys.exit('PRELIMINARY LOGISTIC REGRESSION CONTROL FILE CREATED SUCCESSFULLY')


# RUN_ALL
run_all_generate_preliminary_logistic_regression_cf()

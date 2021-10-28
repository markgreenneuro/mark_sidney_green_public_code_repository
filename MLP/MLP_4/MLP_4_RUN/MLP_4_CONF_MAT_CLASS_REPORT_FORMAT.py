#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd


def add_labels(df: pd.DataFrame, label_columns, label: str = "default"):
    [r, _] = df.shape
    df_labelled = pd.DataFrame(np.repeat(label_columns, r))
    df_labelled.columns = [str(label)]
    df = pd.concat([df_labelled, df], axis=1)
    return df


def class_report_conf_mat(conf_mat, class_report, ni_neurons, nj_neurons, nk_neurons,
                          nl_neurons, epochs, batch_size, optimizer, stutter_test):
    class_report = pd.DataFrame(class_report).transpose()
    class_report = class_report.reset_index()
    class_report = class_report.rename(columns={class_report.columns[0]: 'Key'})
    class_report = add_labels(class_report, ni_neurons, 'NI_NEURONS')
    class_report = add_labels(class_report, nj_neurons, 'NJ_NEURONS')
    class_report = add_labels(class_report, nk_neurons, 'NK_NEURONS')
    class_report = add_labels(class_report, nl_neurons, 'NL_NEURONS')
    class_report = add_labels(class_report, epochs, 'EPOCHS')
    class_report = add_labels(class_report, batch_size, 'BATCH_SIZE')
    class_report = add_labels(class_report, optimizer, 'OPTIMIZER')
    class_report.columns = class_report.columns.str.upper()

    conf_mat = pd.DataFrame(conf_mat)
    for i in range(len(np.unique(stutter_test))):
        conf_mat = conf_mat.rename(index={conf_mat.index[i]: str(np.unique(stutter_test)[i])})
        conf_mat = conf_mat.rename(columns={conf_mat.columns[i]: str(np.unique(stutter_test)[i])})
    conf_mat = conf_mat.reset_index().rename(columns={"index": "CAT"})
    conf_mat = add_labels(conf_mat, ni_neurons, 'NI_NEURONS')
    conf_mat = add_labels(conf_mat, nj_neurons, 'NJ_NEURONS')
    conf_mat = add_labels(conf_mat, nk_neurons, 'NK_NEURONS')
    conf_mat = add_labels(conf_mat, nl_neurons, 'NL_NEURONS')
    conf_mat = add_labels(conf_mat, epochs, 'EPOCHS')
    conf_mat = add_labels(conf_mat, batch_size, 'BATCH_SIZE')
    conf_mat = add_labels(conf_mat, optimizer, 'OPTIMIZER')
    conf_mat.columns = conf_mat.columns.str.upper()

    return [conf_mat, class_report]

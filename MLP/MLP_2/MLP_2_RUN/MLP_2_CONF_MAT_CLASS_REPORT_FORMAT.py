#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd


def add_labels(df: pd.DataFrame, label_columns, label: str = 'default'):
    [r, _] = df.shape
    df_labelled = pd.DataFrame(np.repeat(label_columns, r))
    df_labelled.columns = [str(label)]
    df = pd.concat([df_labelled, df], axis=1)
    return df


def class_report_conf_mat(conf_mat, class_report, ni_neurons, nj_neurons,
                          epochs, batch_size, optimizer):
    classes = np.array(pd.DataFrame(class_report).columns)[:-3]
    classes_df = pd.DataFrame(classes, columns=['CLASSES'])
    class_report = pd.DataFrame(class_report).reset_index()
    class_report = class_report.rename(columns={class_report.columns[0]: 'KEY'})
    class_report.columns = class_report.columns.str.replace(' ', '_')

    class_report = add_labels(class_report, ni_neurons, 'NI_NEURONS')
    class_report = add_labels(class_report, nj_neurons, 'NJ_NEURONS')
    class_report = add_labels(class_report, epochs, 'EPOCHS')
    class_report = add_labels(class_report, batch_size, 'BATCH_SIZE')
    class_report = add_labels(class_report, optimizer, 'OPTIMIZER')

    conf_mat = pd.DataFrame(conf_mat, columns=classes)
    conf_mat = pd.concat([classes_df, conf_mat], axis=1)
    conf_mat = add_labels(conf_mat, ni_neurons, 'NI_NEURONS')
    conf_mat = add_labels(conf_mat, nj_neurons, 'NJ_NEURONS')
    conf_mat = add_labels(conf_mat, epochs, 'EPOCHS')
    conf_mat = add_labels(conf_mat, batch_size, 'BATCH_SIZE')
    conf_mat = add_labels(conf_mat, optimizer, 'OPTIMIZER')
    conf_mat.columns = conf_mat.columns.str.upper()

    return [conf_mat, class_report]

#!/usr/bin/env python
# coding: utf-8
import numpy as np
from os import chdir
import pandas as pd


def add_labels(df: pd.DataFrame, label_columns, label: str = 'default'):
    [r, _] = df.shape
    df_labelled = pd.DataFrame(np.repeat(label_columns, r))
    df_labelled.columns = [str(label)]
    df = pd.concat([df_labelled, df], axis=1)
    return df


def format_loss_accuracy(test_results, ni_neurons, nj_neurons, nk_neurons,
                         epochs, batch_size, optimizer):
    test_results = add_labels(test_results, ni_neurons, 'NI_NEURONS')
    test_results = add_labels(test_results, nk_neurons, 'NK_NEURONS')
    test_results = add_labels(test_results, nj_neurons, 'NJ_NEURONS')
    test_results = add_labels(test_results, epochs, 'EPOCHS')
    test_results = add_labels(test_results, batch_size, 'BATCH_SIZE')
    test_results = add_labels(test_results, optimizer, 'OPTIMIZER')
    test_results.columns = test_results.columns.str.upper()
    return test_results


def output_loss_accuracy(test_results, ni_neurons, nj_neurons, nk_neurons, epochs, batch_size, optimizer, stats_dir):
    name = str(ni_neurons) + '_' + str(nj_neurons) + '_' + str(nk_neurons) + '_' + str(epochs) + '_' + str(
        batch_size) + '_' + str(optimizer)
    name_of_save_file: str = 'LOSS_ACCURACY_' + str(name) + '.csv'
    chdir(stats_dir)
    test_results.to_csv(name_of_save_file, index=False)


def format_and_output_loss_accuracy(test_results, ni_neurons, nj_neurons, nk_neurons, epochs, batch_size, optimizer,
                                    stats_dir):
    test_results = format_loss_accuracy(test_results, ni_neurons, nj_neurons, nk_neurons, epochs, batch_size, optimizer)
    output_loss_accuracy(test_results, ni_neurons, nj_neurons, nk_neurons, epochs, batch_size, optimizer, stats_dir)

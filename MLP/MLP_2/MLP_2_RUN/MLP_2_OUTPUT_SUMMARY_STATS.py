#!/usr/bin/env python
# coding: utf-8
import numpy as np
from os import chdir
import pandas as pd


def output_summary_stats(roc_auc, mse, bic, aic, ni_neurons, nj_neurons, epochs, activation,
                         batch_size, optimizer, num_classes, input_shape, name, stats_dir):
    full_df = pd.concat([roc_auc, mse, bic, aic], axis=1).T.reset_index()
    full_df.columns = ['LABEL', 'VALUE']
    [r, _] = full_df.shape
    roc_auc_df = pd.DataFrame(np.repeat(np.array(roc_auc), r, axis=0), columns=['ROC_AUC'])
    mse_df = pd.DataFrame(np.repeat(np.array(mse), r, axis=0), columns=['MSE'])
    bic_df = pd.DataFrame(np.repeat(np.array(bic), r, axis=0), columns=['BIC'])
    aic_df = pd.DataFrame(np.repeat(np.array(mse), r, axis=0), columns=['AIC'])
    ni_neurons_df = pd.DataFrame(np.repeat(np.array(ni_neurons), r, axis=0), columns=['NI_NEURONS'])
    nj_neurons_df = pd.DataFrame(np.repeat(np.array(nj_neurons), r, axis=0), columns=['NJ_NEURONS'])
    epochs_df = pd.DataFrame(np.repeat(np.array(epochs), r, axis=0), columns=['EPOCHS'])
    activation_df = pd.DataFrame(np.repeat(np.array(activation), r, axis=0), columns=['ACTIVATION'])
    batch_size_df = pd.DataFrame(np.repeat(np.array(batch_size), r, axis=0), columns=['BATCH_SIZE'])
    optimizer_df = pd.DataFrame(np.repeat(np.array(optimizer), r, axis=0), columns=['OPTIMIZER'])
    num_classes_df = pd.DataFrame(np.repeat(np.array(num_classes), r, axis=0), columns=['NUM_CLASSES'])
    input_shape_df = pd.DataFrame(np.repeat(np.array(input_shape), r, axis=0), columns=['INPUT_SIZE'])

    full_df = pd.concat([roc_auc_df, mse_df, bic_df, aic_df, ni_neurons_df, nj_neurons_df,
                         epochs_df, activation_df, batch_size_df, optimizer_df, num_classes_df,
                         input_shape_df, full_df], axis=1)
    full_df = full_df.loc[:, ~full_df.columns.duplicated()]
    name_of_save_file: str = 'SUMMARY_STATS_' + str(name) + '.csv'
    chdir(stats_dir)
    full_df.to_csv(name_of_save_file, index=False)

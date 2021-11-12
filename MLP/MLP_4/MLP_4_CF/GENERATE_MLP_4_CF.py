#!/usr/bin/env python
# coding: utf-8
import numpy as np
from os import chdir
import pandas as pd
from pathlib import Path
import sys


def set_control_file_root():
    control_file_dir = Path('/home/markgreenneuroscience_gmail_com/ML/MLP/MLP_4/MLP_4_CF')
    return control_file_dir


def expand_grid(*args):
    mesh = np.meshgrid(*args)
    return pd.DataFrame(m.flatten() for m in mesh)


def create_mlp_4_grid():
    ni_neurons = np.array([50, 100, 150, 200, 250])
    nj_neurons = np.array([50, 100, 150, 200, 250])
    nk_neurons = np.array([50, 100, 150, 200, 250])
    nl_neurons = np.array([50, 100, 150, 200, 250])
    epochs = np.array([250])
    batch_size = np.array([250])
    optimizer = np.array(['Adam'])
    activation = np.array(['elu'])
    mlp_4 = expand_grid(ni_neurons, nj_neurons, nk_neurons, nl_neurons,
                        epochs, batch_size, optimizer, activation).T
    mlp_4.columns = ['ni_neurons', 'nj_neurons', 'nk_neurons', 'nl_neurons', 'epochs',
                     'batch_size', 'optimizer', 'activation']
    return mlp_4


def run_all_generate_mlp_4_cf() -> None:
    control_file_dir = set_control_file_root()
    mlp_4 = create_mlp_4_grid()
    chdir(control_file_dir)
    mlp_4.to_csv('MLP_4_CF.txt', header=False, index=False, sep='\t')
    sys.exit('MLP 4 CONTROL FILE CREATED SUCCESSFULLY')


# RUN ALL
run_all_generate_mlp_4_cf()

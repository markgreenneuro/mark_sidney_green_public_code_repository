#!/usr/bin/env python
# coding: utf-8
import numpy as np
from os import chdir
import pandas as pd
from pathlib import Path
import sys


def set_control_file_root():
    control_file_dir = Path('/home/markgreenneuroscience_gmail_com/ML/MLP/MLP_2/MLP_2_CF')
    return control_file_dir


def expand_grid(*args):
    mesh = np.meshgrid(*args)
    return pd.DataFrame(m.flatten() for m in mesh)


def create_mlp_2_grid():
    ni_neurons = np.array([50, 100, 150, 200, 250])
    nj_neurons = np.array([50, 100, 150, 200, 250])
    epochs = np.array([250])
    batch_size = np.array([250])
    optimizer = np.array(['Adam'])
    activation = np.array(['elu'])
    mlp_2 = expand_grid(ni_neurons, nj_neurons, epochs, batch_size, optimizer,
                        activation).T
    mlp_2.columns = ['ni_neurons', 'nj_neurons', 'epochs', 'batch_size',
                     'optimizer', 'activation']
    return mlp_2


def run_all_generate_mlp_2_cf() -> None:
    control_file_dir = set_control_file_root()
    mlp_2 = create_mlp_2_grid()
    chdir(control_file_dir)
    mlp_2.to_csv('MLP_2_CF.txt', header=False, index=False, sep='\t')
    sys.exit('MLP 2 CONTROL FILE CREATED SUCCESSFULLY')


# RUN ALL
run_all_generate_mlp_2_cf()

#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os
import shutil


def get_data(datapath):
    os.chdir(datapath)
    DD = pd.read_csv('DATA_DRIVEN_formatted_output.csv', header=None)
    LD = pd.read_csv('LIT_DRIVEN_formatted_output.csv', header=None)
    return [DD, LD]


def create_DD_dir(root, DD):
    os.chdir(root)
    output_data_driven = os.path.join(root, "DATA_DRIVEN")

    if os.path.exists(output_data_driven):
        shutil.rmtree(output_data_driven)
        os.chdir(root)
        os.mkdir(output_data_driven)
        os.chdir(output_data_driven)
        DD.to_csv('DATA_DRIVEN_formatted_output.csv', index=False, header=None)
    else:
        os.chdir(root)
        os.mkdir(output_data_driven)
        os.chdir(output_data_driven)
        DD.to_csv('DATA_DRIVEN_formatted_output.csv', index=False, header=None)
    return output_data_driven


def create_LD_dir(root, LD):
    os.chdir(root)
    output_lit_driven = os.path.join(root, "LIT_DRIVEN")

    if os.path.exists(output_lit_driven):
        shutil.rmtree(output_lit_driven)
        os.chdir(root)
        os.mkdir(output_lit_driven)
        os.chdir(output_lit_driven)
        LD.to_csv('LIT_DRIVEN_formatted_output.csv', index=False, header=None)
    else:
        os.chdir(root)
        os.mkdir(output_lit_driven)
        os.chdir(output_lit_driven)
        LD.to_csv('LIT_DRIVEN_formatted_output.csv', index=False, header=None)
    return output_lit_driven


def format_input_structure(datapath, root):
    [DD, LD] = get_data(datapath)
    output_data_driven = create_DD_dir(root, DD)
    output_lit_driven = create_LD_dir(root, LD)
    return [output_data_driven, output_lit_driven]

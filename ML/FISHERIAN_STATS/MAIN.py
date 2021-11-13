#!/usr/bin/env python
# coding: utf-8
from CREATE_INPUT_FOLDERS import *
from REFORMAT_ASSEMBLE_INDEXES import *
from ONE_SAMPLE_T_TEST import *
from ANOVA_TWO_SAMPLE_T_TEST import *


def get_dirs(root):
    datapath = os.path.join(root, "DATA")
    output_data_driven = os.path.join(root, "DATA_DRIVEN")
    output_lit_driven = os.path.join(root, "LIT_DRIVEN")
    return [datapath, output_data_driven, output_lit_driven]


def run_all_statistics():
    # GET PATHS
    root = "/home/msgreen/Desktop"
    [datapath, output_data_driven, output_data_driven] = get_dirs(root)

    # CREATE_DATA_STRUCTURE
    format_input_structure(datapath, root)

    # DATA_DRIVEN
    data_root = "/home/msgreen/Desktop/DATA_DRIVEN/"
    modality = "DATA_DRIVEN"
    reformat_assemble_indexes(data_root, modality)
    run_one_sample_t_test(data_root, modality)
    run_anova_two_sample_t_tests(data_root, modality)

    # LIT_DRIVEN
    lit_root = "/home/msgreen/Desktop/LIT_DRIVEN/"
    modality = "LIT_DRIVEN"
    reformat_assemble_indexes(lit_root, modality)
    run_one_sample_t_test(lit_root, modality)
    run_anova_two_sample_t_tests(lit_root, modality)


# RUN_ALL
run_all_statistics()

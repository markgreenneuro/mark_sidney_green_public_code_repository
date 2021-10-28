#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd


def output_conf_mat_class_report(name, conf_mat, class_report, stats_dir):
    os.chdir(stats_dir)
    name_of_conf_mat_save_file: str = 'CONF_MAT_' + str(name) + '.csv'
    name_of_class_report_save_file: str = 'CLASS_REPORT_' + str(name) + '.csv'
    pd.DataFrame(conf_mat).to_csv(name_of_conf_mat_save_file, index=False)
    pd.DataFrame(class_report).to_csv(name_of_class_report_save_file, index=False)

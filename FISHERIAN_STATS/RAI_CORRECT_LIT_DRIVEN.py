#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import os
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects.packages import importr

ro.r['options'](warn=-1)
r('as.POSIXct("2015-01-01 00:00:01")+0 ')
base = importr('base')
cars = importr('car')
mvtnorm = importr('mvtnorm')
broom = importr('broom')
psych = importr('psych')
mvtnorm = importr('MHTmult')


def open_formatted_output(root, modality):
    # LOAD_FORMATTED_OUTPUT
    os.chdir(root)
    df = pd.read_csv(str(modality) + '_formatted_output.csv', header=None)
    return df


def create_reformatted_output(df):
    df = pd.DataFrame(np.array(df).ravel().reshape(149, 14))
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    return df


def write_reformatted_output_lit(df, modality):
    # WRITE_REFORMATTED_OUPUT
    df.to_csv(str(modality) + '_reformatted_output.csv', index=None)


def read_reformatted_output_lit(df, modality):
    df = pd.read_csv(str(modality) + '_reformatted_output.csv')
    return df


def remove_bilateral_lit(df):
    df = df[df['Seed'] != 'PPI_bDN6MM_SEED_MFG_BA8_BILAT_RL_ONE']
    df = df[df['Seed'] != 'PPI_bDN6MM_SEED_MFG_BA8_BILAT_RL_TWO']
    df = df[df['Seed'] != 'PPI_bDN6MM_SEED_OFG_BA11_BILAT_RL']
    return df


def correct_reformatted_output_lit(df):
    # DROP AND RENAME
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.rename(columns={"bDN6MM_L_INS_E_-29_9_-9": "bDN6MM_L_INS_E"})
    df = df.rename(columns={"bDN6MM_L_PREC_BA7_BA13_-1_-67_36": "bDN6MM_L_PREC_BA7_BA13"})
    df = df.rename(columns={"bDN6MM_L_PUT_E_-32_2_5": "bDN6MM_L_PUT_E"})
    df = df.rename(columns={"bDN6MM_L_ANT_CING_BA24_BA32_E_-8_37_22": "bDN6MM_L_ANT_CING_BA24_BA32_E"})
    df = df.rename(columns={"bDN6MM_L_PREC_BA13_-4_-60_22": "bDN6MM_L_PREC_BA13"})
    df = df.rename(columns={"bDN6MM_R_INS_E_TWO_41_-8_-9": "bDN6MM_R_INS_E_TWO"})
    df = df.rename(
        columns={"bDN6MM_R_ANG_GYR_BA39_INF_PAR_LOB_BA19_ONE_48_-67_33": "bDN6MM_R_ANG_GYR_BA39_INF_PAR_LOB_BA19_ONE"})
    df = df.rename(
        columns={"bDN6MM_L_ANG_GYR_BA39_INF_PAR_LOB_BA40_-46_-67_44": "bDN6MM_L_ANG_GYR_BA39_INF_PAR_LOB_BA40"})
    df = df.rename(
        columns={"bDN6MM_R_ANG_GYR_BA39_INF_PAR_LOB_BA19_TWO_58_-64_26": "bDN6MM_R_ANG_GYR_BA39_INF_PAR_LOB_BA19_TWO"})
    df = df.rename(columns={"bDN6MM_R_INS_E_ONE_34_12_-6": "bDN6MM_R_INS_E_ONE"})
    df = df.rename(columns={"bDN6MM_L_MID_TEM_GYR_BA21_-64_-29_-6": "bDN6MM_L_MID_TEM_GYR_BA21"})
    df = df.rename(
        columns={"bDN6MM_L_ANG_GYR_BA39_INF_PAR_LOB_BA19_-50_-71_36": "bDN6MM_L_ANG_GYR_BA39_INF_PAR_LOB_BA19"})
    df = df.rename(columns={"bDN6MM_L_INF_PAR_LOB_BA7_-43_-60_58": "bDN6MM_L_INF_PAR_LOB_BA7"})
    df = df.rename(columns={"bDN6MM_L_POST_CING_BA31_-8_-53_30": "bDN6MM_L_POST_CING_BA31"})
    df = df.rename(columns={"bDN6MM_R_CAUD_E_10_5_8": "bDN6MM_R_CAUD_E"})
    df = df.rename(columns={"bDN6MM_R_MID_TEM_GYR_BA21_58_-29_-2": "bDN6MM_R_MID_TEM_GYR_BA21"})
    df = df.rename(columns={"bDN6MM_L_ANG_GYR_BA39_-50_-53_30": "bDN6MM_L_ANG_GYR_BA39"})
    return df


def correct_lit_driven(root, modality):
    df = open_formatted_output(root, modality)
    df = create_reformatted_output(df)
    write_reformatted_output_lit(df, modality)
    df = read_reformatted_output_lit(df, modality)
    df = remove_bilateral_lit(df)
    df = correct_reformatted_output_lit(df)
    write_reformatted_output_lit(df, modality)
    df = read_reformatted_output_lit(df, modality)
    return df

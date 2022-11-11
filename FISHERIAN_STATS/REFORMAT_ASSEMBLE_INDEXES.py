#!/usr/bin/env python
# coding: utf-8
from RAI_CORRECT_DATA_DRIVEN import *
from RAI_CORRECT_LIT_DRIVEN import *
from sklearn.utils.extmath import cartesian
import sys


def get_list(df):
    # GET_LISTS
    # group
    group = pd.DataFrame(df["Group"])
    group = group.drop_duplicates().reset_index(drop=True)
    group.columns = ["Group"]
    # seed
    seed = pd.DataFrame(df["Seed"])
    seed = seed.drop_duplicates().reset_index(drop=True)
    seed.columns = ["Seed"]
    # ROI
    rois = pd.DataFrame(df.columns)
    rois.columns = ["ROIS"]
    rois = rois[~rois['ROIS'].isin(['Group'])]
    rois = rois[~rois['ROIS'].isin(['Seed'])]
    rois = rois[~rois['ROIS'].isin(['Subject_ID'])]
    rois = rois.reset_index(drop=True)
    return [group, seed, rois]


def get_one_sample(group, seed, rois):
    # one_sample\
    one_sample = pd.DataFrame(cartesian((group["Group"], seed["Seed"], rois["ROIS"])))
    one_sample.columns = ["GROUP", "SEED", "ROIS"]
    one_sample = one_sample.dropna().reset_index(drop=True)
    return one_sample


def get_anova(seed, rois):
    # anova
    anova = pd.DataFrame(cartesian((seed["Seed"], rois["ROIS"])))
    anova.columns = ["SEED", "ROIS"]
    anova = anova.dropna().reset_index(drop=True)
    return anova


def get_two_sample(seed, rois):
    # two_sample
    contrast = pd.DataFrame(np.array(["DID-G_v_DID-S", "DID-G_v_CTRL", "DID-S_v_CTRL"]))
    contrast.columns = ["Contrast"]
    two_sample = pd.DataFrame(cartesian((contrast["Contrast"], seed["Seed"], rois["ROIS"])))
    two_sample.columns = ["TEMP", "SEED", "ROI"]
    two_sample_one_two = pd.DataFrame(two_sample.TEMP.str.split('_v_', 1).tolist(), columns=['ONE', 'TWO'])
    del two_sample["TEMP"]
    two_sample = pd.concat([two_sample_one_two, two_sample], axis=1)
    return two_sample


def output_indexer_keys(root, one_sample, anova, two_sample):
    os.chdir(root)
    one_sample.to_csv('ONE_SAMPLE.csv', index=False)
    anova.to_csv('ANOVA.csv', index=False)
    two_sample.to_csv('TWO_SAMPLE.csv', index=False)


def reformat_assemble_indexes(root, modality):
    if modality == "DATA_DRIVEN":
        df = correct_data_driven(root, modality)
    elif modality == "LIT_DRIVEN":
        df = correct_lit_driven(root, modality)
    else:
        sys.exit('MODALITY NOT RECOGNISED.')

    [group, seed, rois] = get_list(df)
    one_sample = get_one_sample(group, seed, rois)
    anova = get_anova(seed, rois)
    two_sample = get_two_sample(seed, rois)
    output_indexer_keys(root, one_sample, anova, two_sample)

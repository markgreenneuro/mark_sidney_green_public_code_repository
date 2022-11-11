#!/usr/bin/env python
# coding: utf-8
from typing import List, Any

import pandas as pd
import numpy as np
import os
import shutil

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
mhtmult = importr('MHTmult')


def get_two_sample_t_tests(rfo, group_set1, group_set2):
    ttest_one = pd.DataFrame([])
    for i in range(len(group_set1)):
        rfo_group_1 = rfo[rfo['GROUP'].isin([group_set1[i]])].reset_index(drop=True)
        rfo_group_2 = rfo[rfo['GROUP'].isin([group_set2[i]])].reset_index(drop=True)
        rfo_group_1 = rfo_group_1["VALUE"]
        rfo_group_2 = rfo_group_2["VALUE"]
        rfo_group_1.to_csv('RFO_GROUP_1.csv', index=False)
        rfo_group_2.to_csv('RFO_GROUP_2.csv', index=False)
        r('TTEST_GROUP_1<-read.csv("RFO_GROUP_1.csv")');
        r('TTEST_GROUP_2<-read.csv("RFO_GROUP_2.csv")');
        r('ttest<-t.test(TTEST_GROUP_1,TTEST_GROUP_2, alternative=c("two.sided","less","greater"),mu=0,paired=FALSE, var.equal=TRUE,conf.level=0.95)')

        ############################################

        group_set1_df = pd.DataFrame([group_set1[i]])
        group_set1_df.columns = ["GROUP_1_NAME"]
        group_set2_df = pd.DataFrame([group_set2[i]])
        group_set2_df.columns = ["GROUP_2_NAME"]

        names_df = pd.concat([group_set1_df, group_set2_df], axis=1)

        ###########################################

        r('describe_osd<-describe(TTEST_GROUP_1)')
        describe_osd = pd.DataFrame(r('describe_osd'))
        [_, c] = describe_osd.shape
        if c > 1:
            describe_osd = describe_osd.T
        else:
            pass
        describe_osd.columns = ['stat']
        group1_mean_df = pd.DataFrame([describe_osd.iloc[2][0]])
        group1_mean_df.columns = ["GROUP_1_MEAN"]
        group1_std_df = pd.DataFrame([describe_osd.iloc[3][0]])
        group1_std_df.columns = ["GROUP_1_STD"]

        r('describe_osd<-describe(TTEST_GROUP_2)')
        describe_osd = pd.DataFrame(r('describe_osd'))
        [_, c] = describe_osd.shape
        if c > 1:
            describe_osd = describe_osd.T
        else:
            pass
        describe_osd.columns = ['STAT']

        group2_mean_df = pd.DataFrame([describe_osd.iloc[2][0]])
        group2_mean_df.columns = ["GROUP_2_MEAN"]
        group2_std_df = pd.DataFrame([describe_osd.iloc[3][0]])
        group2_std_df.columns = ["GROUP_2_STD"]

        mean_std_df = pd.concat([group1_mean_df, group1_std_df, group2_mean_df, group2_std_df], axis=1)

        #########################################

        statdf = pd.DataFrame(r('ttest[1]'))
        statdf.columns = ["TSTAT"]

        pardf = pd.DataFrame(r('ttest[2]'))
        pardf.columns = ["DF"]

        pvaluedf = pd.DataFrame(r('ttest[3]'))
        pvaluedf.columns = ["PVAL"]

        conf_estdf = pd.DataFrame(r('ttest[4]'))
        conf_est_1 = conf_estdf[0]
        conf_est_1 = conf_est_1[0]
        conf_est_df_1 = pd.DataFrame([conf_est_1])
        conf_est_df_1.columns = ["CONF_1"]

        conf_est_2 = conf_estdf[1]
        conf_est_2 = conf_est_2[0]
        conf_est_df_2 = pd.DataFrame([conf_est_2])
        conf_est_df_2.columns = ["CONF_2"]

        stdderdf = pd.DataFrame(r('ttest[7]'))
        stdder = stdderdf[0]
        stdder = stdder[0]
        stdder_df = pd.DataFrame([stdder])
        stdder_df.columns = ["STDERR"]

        statdf_fl = statdf["TSTAT"].round(3).astype(str)
        pardf_fl = pardf["DF"].round(3).astype(str)
        pvaluedf_fl = pvaluedf["PVAL"].round(3).astype(str)
        conf_est_df_1_fl = conf_est_df_1["CONF_1"].round(3).astype(str)
        conf_est_df_2_fl = conf_est_df_2["CONF_2"].round(3).astype(str)

        test_stats_df = pd.concat([statdf, pardf, pvaluedf, conf_est_df_1, conf_est_df_2, stdder_df], axis=1)

        summary_stats_df = "t(" + statdf_fl + ")=" + pardf_fl + ",p=" + pvaluedf_fl + " Conf_interval=[" + conf_est_df_1_fl + "," + conf_est_df_2_fl + "]"
        summary_stats_df = np.array(summary_stats_df)
        summary_stats_df = pd.DataFrame([summary_stats_df])
        summary_stats_df.columns = ["SUMMARY_STATS"]

        #########################################

        ttest = pd.concat([names_df, mean_std_df, test_stats_df, summary_stats_df], axis=1)
        ttest_one = pd.concat([ttest_one, ttest], axis=0)
    ttest_one = pd.DataFrame(ttest_one)
    ttest_one = ttest_one.reset_index(drop=True)
    return ttest_one


def run_stats(root, modality):
    os.chdir(root)
    rfo = pd.read_csv(str(modality) + '_reformatted_output.csv')
    anova_df = pd.read_csv('ANOVA.csv')

    group_set1 = ["DID-G", "DID-G", "DID-S"]
    group_set2 = ["DID-S", "CTRL", "CTRL"]

    did_g_os_seed_roi = pd.read_csv('DID_G_OS_SEED_ROI.csv')
    seed_did_g_os = list(did_g_os_seed_roi["Seed"])
    roi_did_g_os = list(did_g_os_seed_roi["ROI"])
    id_col = ["Group", "Seed", "Subject_ID"]
    roi_did_g_os = [*id_col, *roi_did_g_os]
    rfo = rfo[rfo['Seed'].isin(seed_did_g_os)].reset_index(drop=True)
    rfo = rfo[roi_did_g_os]
    aov_full = pd.DataFrame([])
    ttest_full = pd.DataFrame([])
    [rows, _] = did_g_os_seed_roi.shape

    for i in range(rows):
        ppi_seed = [did_g_os_seed_roi.iloc[i][0]]
        ppi_roi = [did_g_os_seed_roi.iloc[i][1]]
        r.assign('rPPI_SEED', ppi_seed)
        r.assign('rPPI_ROI', ppi_roi)

        id_col = ["Group", "Seed", "Subject_ID"]
        ppi_roi = [*id_col, *ppi_roi]
        rfo = pd.read_csv(str(modality) + '_reformatted_output.csv')
        rfo = rfo[rfo['Seed'].isin(ppi_seed)].reset_index(drop=True)
        rfo = rfo[ppi_roi]
        rfo.columns = ["GROUP", "SEED", "SUBJECT_ID", "VALUE"]
        del rfo["SEED"]
        del rfo["SUBJECT_ID"]
        rfo.to_csv('ANOVA_SELECTED.csv', index=False)
        r('DATAFRAME_NAME<-read.csv("ANOVA_SELECTED.csv")')
        r('DATAFRAME_NAME')
        r('DATAFRAME_NAME$GROUP<-factor(DATAFRAME_NAME$GROUP, levels = c("DID-S", "DID-G", "CTRL"), labels = c("DID-S", "DID-G", "CTRL"))');
        r('y=DATAFRAME_NAME$VALUE');
        r('res.aov <- aov(y ~ GROUP,data= DATAFRAME_NAME)')
        r('res.aov.broom<-data.frame(broom::tidy(res.aov))')
        res_aov = pd.DataFrame(r('res.aov.broom'))
        [_, cols] = res_aov.shape
        if cols < 6:
            res_aov = res_aov.T
        else:
            pass
        res_aov.columns = ["TERM", "DF", "SS", "MS", "STAT", "PVALUE"]
        [res_aov_rows, _] = res_aov.shape
        ppi_seed_col = pd.DataFrame(np.repeat(ppi_seed, res_aov_rows))
        ppi_seed_col.columns = ["PPI"]
        ppi_roi = [anova_df.iloc[i][1]]
        ppi_roi_col = pd.DataFrame(np.repeat(ppi_roi, res_aov_rows))
        ppi_roi_col.columns = ["ROI"]
        res_aov = res_aov.reset_index(drop=True)
        res_aov = pd.concat([ppi_seed_col, ppi_roi_col, res_aov], axis=1)

        ttest_one = get_two_sample_t_tests(rfo, group_set1, group_set2)
        ppi_seed_col = pd.DataFrame(np.repeat(ppi_seed, len(group_set1)))
        ppi_seed_col.columns = ["PPI"]
        ppi_roi = [anova_df.iloc[i][1]]
        ppi_roi_col = pd.DataFrame(np.repeat(ppi_roi, len(group_set1)))
        ppi_roi_col.columns = ["ROI"]
        ttest_one = pd.concat([ppi_seed_col, ppi_roi_col, ttest_one], axis=1)
        ttest_full = pd.concat([ttest_full, ttest_one], axis=0)
        aov_full = pd.concat([aov_full, res_aov], axis=0)
        ttest_full = ttest_full.reset_index(drop=True)
        aov_full = aov_full.reset_index(drop=True)
    return [ttest_full, aov_full]


def perform_gsidak(aov_full):
    p = aov_full["PVALUE"]
    p = pd.DataFrame(p)
    p.to_csv('p.csv', index=False)
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    r('p<-read.csv("p.csv")')
    r('gsidak<-gsidak.p.adjust(p[,1], k=1)')
    r('gsidak')
    gsidak = pd.DataFrame(r('gsidak'))
    gsidak = np.array(gsidak).ravel()
    gsidaknan = np.full((2 * len(gsidak)), np.nan)  #
    gsidaknan[::2] = gsidak
    gsidaknan = pd.DataFrame(gsidaknan)
    gsidaknan.columns = ["PVAL_ADJUST"]
    gsidaknan = gsidaknan.reset_index(drop=True)
    aov_full = pd.concat([aov_full, gsidaknan], axis=1)
    aov_full = aov_full.loc[:, ~aov_full.columns.duplicated()]
    return aov_full


def bonferroni(ttest_full):
    pvalslist = ttest_full["PVAL"]
    pvalslist = pvalslist * 3
    pvalslist[pvalslist > 1] = 1
    pvalslist = pd.DataFrame(pvalslist)
    pvalslist.columns = ["PVAL_BON"]
    ttest_full = pd.concat([ttest_full, pvalslist], axis=1)
    ttest_full = ttest_full.loc[:, ~ttest_full.columns.duplicated()]
    return ttest_full


def select_significant(ttest_full, aov_full):
    ttest_full_pval = ttest_full.loc[ttest_full['PVAL'] < 0.05].reset_index(drop=True)
    ttest_full_pval_bon = ttest_full.loc[ttest_full['PVAL_BON'] < 0.05].reset_index(drop=True)
    aov_full_pval = aov_full.loc[aov_full['PVAL_ADJUST'] < 0.05].reset_index(drop=True)
    return [ttest_full_pval, ttest_full_pval_bon, aov_full_pval]


def filter_significant_t(aov_full_pval, ttest_full):
    [aov_full_pval_rows, _] = aov_full_pval.shape
    ttest_full_sig_anova_full = pd.DataFrame([])
    for i in range(aov_full_pval_rows):
        seed = aov_full_pval["PPI"].iloc[i]
        roi = aov_full_pval["ROI"].iloc[i]
        ttest_full_sig_anova = ttest_full[(ttest_full["PPI"] == seed) & (ttest_full["ROI"] == roi)].dropna()
        ttest_full_sig_anova = ttest_full_sig_anova.reset_index(drop=True)
        ttest_full_sig_anova_full = pd.concat([ttest_full_sig_anova, ttest_full_sig_anova_full], axis=0)
    ttest_full_sig_anova_full = ttest_full_sig_anova_full.reset_index(drop=True)
    return ttest_full_sig_anova_full


def create_output_dir(root, modality):
    os.chdir(root)
    outputdir = str(modality) + '_PPI_OUTPUT_TWO_SIDED_STATS'
    outputdirpath = os.path.join(root, outputdir)

    if os.path.exists(outputdirpath):
        shutil.rmtree(outputdirpath)  # THIS THROWS ERROR IF N/EMPTY
        os.chdir(root)
        os.mkdir(outputdir)
        os.chdir(outputdir)

    else:
        os.chdir(root)
        os.mkdir(outputdir)
        os.chdir(outputdir)
    return outputdir


def write_output_part_1(root, outputdir, modality, aov_full, ttest_full):
    os.chdir(root)
    os.chdir(outputdir)
    name = str(modality) + '_PPI_OUTPUT_TWO_SIDED_STATS_OUTPUT.xlsx'
    with pd.ExcelWriter(name, engine='openpyxl') as writer:
        aov_full.to_excel(writer, sheet_name='ANOVA')
        ttest_full.to_excel(writer, sheet_name='TTEST')


def write_output_part_2(root, outputdir, modality, aov_full_pval, ttest_full_sig_anova_full):
    os.chdir(root)
    os.chdir(outputdir)
    name = str(modality) + '_PPI_OUTPUT_TWO_SIDED_STATS_OUTPUT_SIGNIFICANT.xlsx'
    with pd.ExcelWriter(name, engine='openpyxl') as writer:
        aov_full_pval.to_excel(writer, sheet_name='ANOVA_SIG')
        ttest_full_sig_anova_full.to_excel(writer, sheet_name='TTEST_ANOVA_SELECTED')


def remove_excess_csv(root):
    os.chdir(root)
    os.remove("ANOVA_SELECTED.csv")
    os.remove("RFO_GROUP_1.csv")
    os.remove("RFO_GROUP_2.csv")


def run_anova_two_sample_t_tests(root, modality):
    [ttest_full, aov_full] = run_stats(root, modality)
    ttest_full = bonferroni(ttest_full)
    aov_full = perform_gsidak(aov_full)
    aov_full = aov_full.reset_index(drop=True)
    [ttest_full_pval, ttest_full_pval_bon, aov_full_pval] = select_significant(ttest_full, aov_full)
    ttest_full_sig_anova_full = filter_significant_t(aov_full_pval, ttest_full)
    outputdir = create_output_dir(root, modality)
    aov_full = aov_full.round(3)
    ttest_full = ttest_full.round(3)
    aov_full_pval = aov_full_pval.round(3)
    ttest_full = ttest_full.round(3)
    ttest_full_sig_anova_full = ttest_full_sig_anova_full.round(3)
    write_output_part_1(root, outputdir, modality, aov_full, ttest_full)
    write_output_part_2(root, outputdir, modality, aov_full_pval, ttest_full_sig_anova_full)
    remove_excess_csv(root)

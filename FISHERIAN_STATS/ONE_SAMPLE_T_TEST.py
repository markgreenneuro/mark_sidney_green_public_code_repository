#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import os
import shutil
import openpyxl
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


def get_rfo_os_num_rows(root, modality):
    os.chdir(root)
    rfo = pd.read_csv(str(modality) + '_reformatted_output.csv')
    ones = pd.read_csv('ONE_SAMPLE.csv')
    [num_rows, _] = ones.shape
    return [rfo, ones, num_rows]


def run_stats(ones, rfo, row):
    group = ones.iloc[row]["GROUP"]
    seed = ones.iloc[row]["SEED"]
    rois = ones.iloc[row]["ROIS"]
    r.assign('rGROUP', group)
    r.assign('rSEED', seed)
    r.assign('rROIS', rois)

    rfo_gs = rfo.loc[rfo['Group'] == group].reset_index(drop=True)
    rfo_gs_rs = pd.concat([rfo_gs[["Group", "Seed", "Subject_ID"]], rfo_gs[rois]], axis=1)
    rfo_gs_rs_ss = rfo_gs_rs.loc[rfo_gs_rs['Seed'] == seed].reset_index(drop=True)
    osd = pd.DataFrame(rfo_gs_rs_ss[rois])
    osd.to_csv('osd.csv', index=False)
    ro.globalenv['osd'] = "osd.csv"
    r('osd<-read.csv(osd)')
    r('ttest <- t.test(osd, mu = 0)')

    roisdf = pd.DataFrame([rois])
    roisdf.columns = ["ROI"]
    seeddf = pd.DataFrame([seed])
    seeddf.columns = ["Seed"]
    groupdf = pd.DataFrame([group])
    groupdf.columns = ["Group"]
    name = pd.concat([groupdf, seeddf, roisdf], axis=1)

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

    summary_stats = "t(" + statdf_fl + ")=" + pardf_fl + ",p=" + pvaluedf_fl + " Conf_interval=[" + conf_est_df_1_fl + "," + conf_est_df_2_fl + "]"
    summary_stats = np.array(summary_stats)
    summary_stats = pd.DataFrame([summary_stats])
    summary_stats.columns = ["summary_stats"]

    r('describe_osd<-describe(osd)')

    describe_osd = pd.DataFrame(r('describe_osd'))
    [_, c] = describe_osd.shape
    if c > 1:
        describe_osd = describe_osd.T
    else:
        pass
    describe_osd.columns = ['STAT']
    mean_df = pd.DataFrame([describe_osd.iloc[2][0]])
    mean_df.columns = ["MEAN"]
    std_df = pd.DataFrame([describe_osd.iloc[3][0]])
    std_df.columns = ["STD"]

    onesample_t = pd.concat([name, mean_df, std_df, pardf, statdf, pvaluedf, conf_est_df_1, conf_est_df_2], axis=1)
    return onesample_t


def get_full_onesample(root, modality):
    [rfo, ones, num_rows] = get_rfo_os_num_rows(root, modality)
    full_onesample_t = pd.DataFrame([])
    for row in range(num_rows):
        onesample_t_df = run_stats(ones, rfo, row)
        full_onesample_t = pd.concat([full_onesample_t, onesample_t_df], axis=0)
    full_onesample_t = full_onesample_t.reset_index(drop=True)
    return full_onesample_t


def calculate_pval_bon(df):
    p = df["PVAL"]
    p = pd.DataFrame(p)
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    ro.globalenv['p'] = p
    r('p<-cbind(p)');
    r('p<-unlist(p)');
    r('pval_adjust<-gsidak.p.adjust(p, k=1)')
    pval_adjust = pd.DataFrame(r('pval_adjust'))
    pval_adjust.columns = ["PVAL_ADJUST"]
    df = pd.concat([df, pval_adjust], axis=1)
    return df


def run_split_and_bonferroni(root, modality):
    full_onesample_t_df = get_full_onesample(root, modality)
    did_g = full_onesample_t_df.loc[full_onesample_t_df['Group'] == 'DID-G'].reset_index(drop=True)
    did_s = full_onesample_t_df.loc[full_onesample_t_df['Group'] == 'DID-S'].reset_index(drop=True)
    ctrl = full_onesample_t_df.loc[full_onesample_t_df['Group'] == 'ctrl'].reset_index(drop=True)
    full_onesample_t_df = calculate_pval_bon(full_onesample_t_df)
    did_g = calculate_pval_bon(did_g)
    did_s = calculate_pval_bon(did_s)
    ctrl = calculate_pval_bon(ctrl)
    return [full_onesample_t_df, did_g, did_s, ctrl]


def p_val_cut_off(df, col):
    df_cut = df.loc[df[col] < 0.05]
    df_cut = df_cut.reset_index(drop=True)
    return df_cut


def check_rows_len(df):
    [rows, _] = df.shape
    return rows


def create_output_dir_file(root, modality):
    os.chdir(root)
    outputdir = str(modality) + '_PPI_ONE_SAMPLE'
    outputdir = os.path.join(root, outputdir)

    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)  # THIS THROWS ERROR IF N/EMPTY
        os.chdir(root)
        os.mkdir(outputdir)
        os.chdir(outputdir)

        wb = openpyxl.Workbook()
        outfile_name = str(modality) + '_OS_OUTPUT_FULL.xlsx'
        outputfile_path = os.path.join(root, outputdir, outfile_name)
        wb.save(outputfile_path)

    else:
        os.chdir(root)
        os.mkdir(outputdir)
        os.chdir(outputdir)

        wb = openpyxl.Workbook()
        outfile_name = str(modality) + '_OS_OUTPUT_FULL.xlsx'
        outputfile_path = os.path.join(root, outputdir, outfile_name)
        wb.save(outputfile_path)
    return outputdir


def write_output(outputdir, modality, full_onesample_t, did_g, did_s, ctrl):
    os.chdir(outputdir)
    outfile_name = str(modality) + '_OS_OUTPUT_FULL.xlsx'
    with pd.ExcelWriter(outfile_name, engine='openpyxl') as writer:
        full_onesample_t.to_excel(writer, sheet_name='FULL_HS')

        did_g.to_excel(writer, sheet_name='DID_G')
        did_s.to_excel(writer, sheet_name='DID_S')
        ctrl.to_excel(writer, sheet_name='CTRL')

        full_onesample_t_hs = p_val_cut_off(full_onesample_t, 'PVAL_ADJUST')
        did_g_hs = p_val_cut_off(did_g, 'PVAL_ADJUST')
        did_s_hs = p_val_cut_off(did_s, 'PVAL_ADJUST')
        ctrl_hs = p_val_cut_off(ctrl, 'PVAL_ADJUST')

        if check_rows_len(full_onesample_t_hs) != 0:
            full_onesample_t_hs.to_excel(writer, sheet_name='FULL_G_HS')
        if check_rows_len(did_g_hs) != 0:
            did_g_hs.to_excel(writer, sheet_name='DID_G_HS')
        if check_rows_len(did_s_hs) != 0:
            did_s_hs.to_excel(writer, sheet_name='DID_S_HS')
        if check_rows_len(ctrl_hs) != 0:
            ctrl_hs.to_excel(writer, sheet_name='CTRL_HS')

        full_onesample_t_p = p_val_cut_off(full_onesample_t, 'PVAL')
        did_g_p = p_val_cut_off(did_g, 'PVAL')
        did_s_p = p_val_cut_off(did_s, 'PVAL')
        ctrl_p = p_val_cut_off(ctrl, 'PVAL')

        if check_rows_len(full_onesample_t_hs) != 0:
            full_onesample_t_p.to_excel(writer, sheet_name='FULL_G_P')
        if check_rows_len(did_g_p) != 0:
            did_g_p.to_excel(writer, sheet_name='DID_G_P')
        if check_rows_len(did_s_p) != 0:
            did_s_p.to_excel(writer, sheet_name='DID_S_P')
        if check_rows_len(ctrl_p) != 0:
            ctrl_p.to_excel(writer, sheet_name='CTRL_P')


def save_did_g_pval_list(root, did_g):
    os.chdir(root)
    did_g_pval = did_g.loc[did_g['PVAL'] < 0.05].reset_index(drop=True)
    did_g_pval_list = did_g_pval[["Seed", "ROI"]]
    did_g_pval_list = did_g_pval_list.drop_duplicates()
    did_g_pval_list = pd.DataFrame(did_g_pval_list)
    did_g_pval_list.columns = ["Seed", "ROI"]
    did_g_pval_list.to_csv('DID_G_OS_SEED_ROI.csv', index=False)


def remove_osd_csv(root):
    os.remove("osd.csv")


def run_one_sample_t_test(root, modality):
    [full_onesample_t, did_g, did_s, ctrl] = run_split_and_bonferroni(root, modality)
    outputdir = create_output_dir_file(root, modality)
    full_onesample_t = full_onesample_t.round(3)
    did_g = did_g.round(3)
    did_s = did_s.round(3)
    ctrl = ctrl.round(3)
    write_output(outputdir, modality, full_onesample_t, did_g, did_s, ctrl)
    save_did_g_pval_list(root, did_g)
    remove_osd_csv(root)

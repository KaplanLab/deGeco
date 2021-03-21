import warnings
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib
import itertools
from scipy import stats
import subprocess

import gc_model as gc
import array_utils
import hic_analysis as hic
from toolz.curried import *

fit2mat = lambda fit: gc.generate_interactions_matrix(**fit)

batch_load = compose(map(np.load), glob.glob)
downsampled_chr19 = lambda res, pct: np.load(f'/storage/md_kaplan/hagaik/paper/figure2/downsampled_data_noallzeros/chr19_{res}_{float(pct)}.npy')
downsampled_fit = lambda res, pct, st: np.load(f'/storage/md_kaplan/hagaik/paper/figure2/downsampled_data_noallzeros/fit_diag_{st}st_chr19_{res}_{float(pct)}_best.npz', allow_pickle=True)['parameters'][()]

st = 2
resolutions = (5000, 10000, 20000)
sample_rates = (0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1)

@curry
def likelihood(interactions_mat, fit_mat, mask=True):
    clean_interactions_mat = array_utils.normalize_tri_l1(gc.preprocess(interactions_mat))
    unique_interactions = array_utils.get_lower_triangle(clean_interactions_mat)
    clean_fit_mat = array_utils.normalize_tri_l1(gc.preprocess(fit_mat))
    unique_fit_interactions = array_utils.get_lower_triangle(clean_fit_mat)
    log_unique_fit_interactions = np.log(unique_fit_interactions)
    nn = np.isfinite(unique_interactions) & np.isfinite(log_unique_fit_interactions) & mask

    return np.sum(unique_interactions[nn] * log_unique_fit_interactions[nn])

@curry
def likelihood_ratio(interactions_mat, fit_mat):
    nz = True #array_utils.get_lower_triangle((interactions_mat != 0) & (fit_mat != 0), k=-1)
    return np.exp(likelihood(interactions_mat, fit_mat, nz) - likelihood(interactions_mat, interactions_mat, nz))

@curry
def calc_pearsonr(mat1, mat2):
    nn = np.isfinite(mat1) & np.isfinite(mat2)
    return stats.pearsonr(mat1[nn], mat2[nn])[0]

@curry
def calc_spearmanr(mat1, mat2):
    nn = np.isfinite(mat1) & np.isfinite(mat2)
    return stats.spearmanr(mat1[nn], mat2[nn])[0]

def calc_performance_stats(st, resolutions, sample_rates):
    pearson_corrs = np.empty((len(resolutions), len(sample_rates)))
    spearman_corrs = np.empty((len(resolutions), len(sample_rates)))
    lambda_mean_diffs = np.empty((len(resolutions), len(sample_rates)))
    lr = np.empty((len(resolutions), len(sample_rates)))
    normalized_pearsonr = np.empty((len(resolutions), len(sample_rates)))
    for i, res in enumerate(resolutions):
        print(f"* Resolution: {res}")
        fit_all_reads = downsampled_fit(res, 1, st)
        input_mat = downsampled_chr19(res, 1)
        for j, pct in enumerate(sample_rates):
            print(f"** Sample rate: {pct:.1%}")
            print("*** Loading matrices:", end=' ')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                fit = downsampled_fit(res, pct, st)   
                fitted_mat = fit2mat(fit)
                normalized_input = hic.normalize_distance(input_mat)
                normalized_fit = hic.normalize_distance(fitted_mat)
            print("done")

            print("*** Pearson:", end=' ')
            pearson_corrs[i, j] = calc_pearsonr(input_mat, fitted_mat)
            print(pearson_corrs[i, j])
            print("*** Spearman:", end=' ')
            spearman_corrs[i, j] = calc_spearmanr(input_mat, fitted_mat)
            print(spearman_corrs[i, j])
            print("*** Lambda mean diff:", end=' ')
            lambda_mean_diffs[i, j] = np.minimum(
                                        np.nanmean(np.abs(fit['state_probabilities'][:, 0] - fit_all_reads['state_probabilities'][:, 0])),
                                        np.nanmean(np.abs(fit['state_probabilities'][:, 0] - fit_all_reads['state_probabilities'][:, 1]))
                                    )
            print(lambda_mean_diffs[i, j])
            print("*** Normalized pearson:", end=' ')
            normalized_pearsonr[i, j] = calc_pearsonr(normalized_input, normalized_fit)
            print(normalized_pearsonr[i, j])
            print("*** LR:", end=' ')
            lr[i, j] = likelihood_ratio(input_mat, fitted_mat)
            print(lr[i, j])
    return dict(pearson_corrs=pearson_corrs, spearman_corrs=spearman_corrs, lambda_mean_diffs=lambda_mean_diffs,
               normalized_pearson_corrs=normalized_pearsonr, lr=lr)

def calc_optimal_stats_sampled_fit_vs_resampled(st, resolutions, sample_rates):
    pearson_corrs = np.empty((len(resolutions), len(sample_rates)))
    spearman_corrs = np.empty((len(resolutions), len(sample_rates)))
    lr = np.empty((len(resolutions), len(sample_rates)))
    normalized_pearsonr = np.empty((len(resolutions), len(sample_rates)))
    for i, res in enumerate(resolutions):
        print(f"* Resolution: {res}")
        for j, pct in enumerate(sample_rates):
            print(f"** Sample rate: {pct:.1%}")
            print("** Loading fit:", end=" ")
            fitted_mat_all_reads = fit2mat(downsampled_fit(res, pct, st))
            print("done")
            simulated_mats = batch_load(f'/srv01/technion/hagaik/storage/paper/figure2/resampled_data_noallzeros/chr19_{st}st_{res}_{float(pct)}_run*.npy')
            print('*** Calculating correlations')
            get_stats = curry(juxt(calc_pearsonr,
                             calc_spearmanr,
                             likelihood_ratio,
                             lambda x, y: calc_pearsonr(hic.normalize_distance(x), hic.normalize_distance(y))
                            ))
            stats = list(map(get_stats(fitted_mat_all_reads), simulated_mats))
            pearson_corrs[i, j], spearman_corrs[i, j], lr[i, j], normalized_pearsonr[i, j] = np.mean(stats, axis=0)
            print("*** Mean pearson:", pearson_corrs[i, j])
            print("*** Mean spearman:", spearman_corrs[i, j])
            print("*** Mean LR:", lr[i, j])
            print("*** Mean normalized pearson:", normalized_pearsonr[i, j])

    return dict(pearson_corrs=pearson_corrs, spearman_corrs=spearman_corrs,
                normalized_pearson_corrs=normalized_pearsonr, lr=lr)

performance_stats = calc_performance_stats(st, resolutions, sample_rates)
np.savez('/srv01/technion/hagaik/storage/paper/figure2/performance_stats_arrow_C.npz', **performance_stats)

optimal_stats = calc_optimal_stats_sampled_fit_vs_resampled(st, resolutions, sample_rates)
np.savez('/srv01/technion/hagaik/storage/paper/figure2/optimal_stats_arrow_B.npz', **optimal_stats)

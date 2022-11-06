import warnings
import glob
import sys
import itertools
from scipy import stats
import numpy as np

import gc_model as gc
import gc_datafile
import array_utils
import hic_analysis as hic
from toolz.curried import *

batch_load = compose(map(np.load), glob.glob)
downsampled_chr19 = lambda res, pct: hic.get_matrix_from_coolfile(f'/storage/md_kaplan/hagaik/chr19_downsampled/chr19_downsampled_{float(pct)}.mcool', res, 'all')
downsampled_fit = lambda res, pct, st: gc_datafile.load_params(f'/storage/md_kaplan/hagaik/chr19_downsampled/{st}states/output/fit_{st}st_chr19_{res}_{float(pct)}_best.npz')

resolutions = (10000, 20000, 100000, 500000)
sample_rates = (0.0005,0.001,0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1) # 0.001 and below are all-NaNs

@curry
def calc_pearsonr(mat1, mat2):
    nn = np.isfinite(mat1) & np.isfinite(mat2)
    return stats.pearsonr(mat1[nn], mat2[nn])[0]

@curry
def calc_spearmanr(mat1, mat2):
    nn = np.isfinite(mat1) & np.isfinite(mat2)
    return stats.spearmanr(mat1[nn], mat2[nn])[0]

@curry
def likelihood(interactions_mat, fit_mat, mask=True):
    clean_interactions_mat = array_utils.normalize_tri_l1(gc.preprocess(interactions_mat))
    unique_interactions = array_utils.get_lower_triangle(clean_interactions_mat)
    clean_fit_mat = array_utils.normalize_tri_l1(gc.preprocess(fit_mat))
    unique_fit_interactions = array_utils.get_lower_triangle(clean_fit_mat)
    log_unique_fit_interactions = np.log(unique_fit_interactions)
    nn = np.isfinite(unique_interactions) & np.isfinite(log_unique_fit_interactions) & mask

    return np.sum(unique_interactions[nn] * log_unique_fit_interactions[nn])

def calc_lambda_diff(states1, states2):
    mean_diffs = [ np.nanmean(np.abs(states1.T - p)) for p in itertools.permutations(states2.T) ]
    return np.min(mean_diffs)

@curry
def likelihood_ratio(interactions_mat, fit_mat):
    nz = array_utils.get_lower_triangle((interactions_mat != 0) & (fit_mat != 0), k=-1)
    return np.exp(likelihood(interactions_mat, fit_mat, nz) - likelihood(interactions_mat, interactions_mat, nz))

def calc_performance_stats(st, resolutions, sample_rates):
    pearson_corrs = np.empty((len(resolutions), len(sample_rates)))
    spearman_corrs = np.empty((len(resolutions), len(sample_rates)))
    lambda_mean_diffs = np.empty((len(resolutions), len(sample_rates)))
    lr = np.empty((len(resolutions), len(sample_rates)))
    normalized_pearsonr = np.empty((len(resolutions), len(sample_rates)))
    normalized_spearmanr = np.empty((len(resolutions), len(sample_rates)))
    for i, res in enumerate(resolutions):
        print(f"* Resolution: {res}")
        fit_all_reads = downsampled_fit(res, 1, st)
        input_mat = downsampled_chr19(res, 1)
        normalized_input = hic.normalize_distance(input_mat)
        for j, pct in enumerate(sample_rates):
            print(f"** Sample rate: {pct:.2%}")
            print("*** Loading matrices:", end=' ')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                fit = downsampled_fit(res, pct, st)   
                fitted_mat = gc.generate_interactions_matrix(**fit)
                normalized_fit = hic.normalize_distance(fitted_mat)
            print("done")

            print("*** Pearson:", end=' ')
            pearson_corrs[i, j] = calc_pearsonr(input_mat, fitted_mat)
            print(pearson_corrs[i, j])
            print("*** Spearman:", end=' ')
            spearman_corrs[i, j] = calc_spearmanr(input_mat, fitted_mat)
            print(spearman_corrs[i, j])
            print("*** Lambda mean diff:", end=' ')
            lambda_mean_diffs[i, j] = calc_lambda_diff(fit['state_probabilities'], fit_all_reads['state_probabilities'])
            print(lambda_mean_diffs[i, j])
            print("*** Normalized pearson:", end=' ')
            normalized_pearsonr[i, j] = calc_pearsonr(normalized_input, normalized_fit)
            print(normalized_pearsonr[i, j])
            print("*** Normalized spearman:", end=' ')
            normalized_spearmanr[i, j] = calc_spearmanr(normalized_input, normalized_fit)
            print(normalized_spearmanr[i, j])            
            print("*** LR:", end=' ')
            lr[i, j] = likelihood_ratio(input_mat, fitted_mat)
            print(lr[i, j])
    return dict(pearson_corrs=pearson_corrs, spearman_corrs=spearman_corrs, lambda_mean_diffs=lambda_mean_diffs,
               normalized_pearson_corrs=normalized_pearsonr, normalized_spearman_corrs=normalized_spearmanr,
                lr=lr)

if __name__ == '__main__':
    if 1:
        try:
            st = int(sys.argv[1])
        except:
            if "--help" in sys.argv:
                print(f"Usage: {sys.argv[0]} [STATES]")
                sys.exit(0)
            st = 2
        print(f"Calculating performance stats for states={st}")
        performance_stats = calc_performance_stats(st, resolutions, sample_rates)
        np.savez(f'/srv01/technion/hagaik/storage/chr19_downsampled/{st}states/performance_stats_vs_allreads.npz', **performance_stats)

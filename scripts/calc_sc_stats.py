import warnings
import os
import glob
import itertools
import sys

import numpy as np
from scipy import stats
from toolz.curried import *

sys.path.append('.')
import gc_model as gc
from gc_datafile import load_params
import array_utils
import hic_analysis as hic

run_dir = sys.argv[1]
fit_to_mat = lambda fit: gc.generate_interactions_matrix(**fit)
sc_hic = lambda s, res, reads: load_params(f"{run_dir}/fit/chr19_{s}st_res{res}_reads{reads}_best.npz")
orig = lambda s, res: load_params(f"{run_dir}/origin/chr19_{s}st_{res}_best.npz")

st = 2
resolutions = [40000, 50000, 100000, 250000, 500000, 1000000]
reads = [100000, 250000, 500000, 750000, 1000000]

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
    nz = array_utils.get_lower_triangle(fit_mat != 0, k=-1)
    return np.exp(likelihood(interactions_mat, fit_mat, nz) - likelihood(interactions_mat, interactions_mat, nz))

@curry
def calc_pearsonr(mat1, mat2):
    nn = np.isfinite(mat1) & np.isfinite(mat2)
    return stats.pearsonr(mat1[nn], mat2[nn])[0]

@curry
def calc_spearmanr(mat1, mat2):
    nn = np.isfinite(mat1) & np.isfinite(mat2)
    return stats.spearmanr(mat1[nn], mat2[nn])[0]

def gc_fit(fit):
    gc = fit['state_probabilities'] @ fit['cis_weights'] @ fit['state_probabilities'].T
    np.fill_diagonal(gc, np.nan)
    return array_utils.normalize_tri_l1(gc)


def calc_sc_stats(st, resolutions, reads):
    pearson_corrs = np.empty((len(resolutions), len(reads)))
    spearman_corrs = np.empty((len(resolutions), len(reads)))
    lambda_mean_diffs = np.empty((len(resolutions), len(reads)))
    lr = np.empty((len(resolutions), len(reads)))
    normalized_pearsonr = np.empty((len(resolutions), len(reads)))
    normalized_spearmanr = np.empty((len(resolutions), len(reads)))
    parametric_normalized_pearsonr = np.empty((len(resolutions), len(reads)))
    parametric_normalized_spearmanr = np.empty((len(resolutions), len(reads)))
    gc_pearson = np.empty((len(resolutions), len(reads)))
    gc_spearman = np.empty((len(resolutions), len(reads)))
    for i, res in enumerate(resolutions):
        print(f"* Resolution: {res}")
        print("** Loading original fit:", end=" ")
        orig_fit = orig(st, res)
        orig_mat = fit_to_mat(orig_fit)
        orig_normalized = hic.normalize_distance(orig_mat)
        normalizing_mat = orig_normalized / orig_mat
        print("done")
        for j, r in enumerate(reads):
            print(f"** Reads: {r}")
            print(f"** Loading fit:", end=" ")
            sc_fit = sc_hic(st, res, r)
            sc_mat = gc.generate_interactions_matrix(**sc_fit)
            sc_normalized = hic.normalize_distance(sc_mat)
            sc_normalized_parametric = sc_mat * normalizing_mat
            print("done")
            print("*** Pearson:", end=' ')
            pearson_corrs[i, j] = calc_pearsonr(orig_mat, sc_mat)
            print(pearson_corrs[i, j])
            print("*** Spearman:", end=' ')
            spearman_corrs[i, j] = calc_spearmanr(orig_mat, sc_mat)
            print(spearman_corrs[i, j])
            print("*** Lambda mean diff:", end=' ')
            lambda_mean_diffs[i, j] = np.minimum(
                                        np.nanmean(np.abs(sc_fit['state_probabilities'][:, 0] - orig_fit['state_probabilities'][:, 0])),
                                        np.nanmean(np.abs(sc_fit['state_probabilities'][:, 0] - orig_fit['state_probabilities'][:, 1]))
                                    )
            print(lambda_mean_diffs[i, j])
            print("*** Normalized pearson:", end=' ')
            normalized_pearsonr[i, j] = calc_pearsonr(orig_normalized, sc_normalized)
            print(normalized_pearsonr[i, j])
            print("*** Normalized spearman:", end=' ')
            normalized_spearmanr[i, j] = calc_spearmanr(orig_normalized, sc_normalized)
            print(normalized_spearmanr[i, j])
            print("*** Normalized pearson (parameteric):", end=' ')
            parametric_normalized_pearsonr[i, j] = calc_pearsonr(orig_normalized, sc_normalized_parametric)
            print(parametric_normalized_pearsonr[i, j])
            print("*** Normalized spearman (parameteric):", end=' ')
            parametric_normalized_spearmanr[i, j] = calc_spearmanr(orig_normalized, sc_normalized_parametric)
            print(parametric_normalized_spearmanr[i, j])
            print("*** LR:", end=' ')
            lr[i, j] = likelihood_ratio(orig_mat, sc_mat)
            print(lr[i, j])
            print("*** GC-only pearson", end=' ')
            gc_pearson[i, j] = calc_pearsonr(gc_fit(orig_fit), gc_fit(sc_fit))
            print(gc_pearson[i, j])
            print("*** GC-only spearman", end=' ')
            gc_spearman[i, j] = calc_spearmanr(gc_fit(orig_fit), gc_fit(sc_fit))
            print(gc_spearman[i, j])
    return dict(pearson_corrs=pearson_corrs, spearman_corrs=spearman_corrs,
                normalized_pearson_corrs=normalized_pearsonr, normalized_spearman_corrs=normalized_spearmanr,
                parametric_normalized_pearson_corrs=parametric_normalized_pearsonr, parametric_normalized_spearman_corrs=parametric_normalized_spearmanr,
                lr=lr, lambda_mean_diffs=lambda_mean_diffs, gc_pearson=gc_pearson, gc_spearman=gc_spearman)

print(f"Using run dir: {run_dir}")
sc_stats = calc_sc_stats(2, resolutions, reads)
output_file=f"{run_dir}/stats_best.npz"
print("Saving to file", output_file)
np.savez(output_file, **sc_stats)
print("Done")

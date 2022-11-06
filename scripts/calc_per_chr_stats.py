import warnings
import glob
import sys

from scipy import stats
import numpy as np
from toolz.curried import *

sys.path.append('.')
import gc_model as gc
import gc_datafile
import array_utils
import hic_analysis as hic

mcool_filename = '/srv01/technion/hagaik/storage/Rao_GM12878_zoomified.mcool'

def cis_trans_mask_slice(cis_lengths, slice_x, slice_y):
    groups = np.arange(np.size(cis_lengths))
    groups_per_bin = np.repeat(groups, cis_lengths)
    groups_per_bin_x = groups_per_bin[slice_x]
    groups_per_bin_y = groups_per_bin[slice_y]
    return groups_per_bin_x[:, None] == groups_per_bin_y[None, :]

def log_gc_interactions_slice(lambdas, cis_weights, trans_weights, cis_trans_mask, slice_x,
                              slice_y):
    lambdas_x = lambdas[slice_x]
    lambdas_y = lambdas[slice_y]
    cis_interactions = lambdas_x @ cis_weights @ lambdas_y.T
    trans_interactions = lambdas_x @ trans_weights @ lambdas_y.T
    combined_interactions = np.where(cis_trans_mask, cis_interactions, trans_interactions)

    return np.log(combined_interactions)

def log_dd_interactions_slice(alpha, beta, n, cis_trans_mask, slice_x, slice_y):
    bin_distances = np.arange(n)
    bin_distances_x = bin_distances[slice_x]
    bin_distances_y = bin_distances[slice_y]
    distances = 1.0 * np.abs(bin_distances_x[:, None] - bin_distances_y[None, :])
    distances[distances == 0] = np.nan # Remove main diag
    cis_interactions = alpha * np.log(distances)
    trans_interactions = np.full_like(cis_interactions, beta)
    
    return np.where(cis_trans_mask, cis_interactions, trans_interactions)

def generate_interactions_mat_slice(slice_x, slice_y, state_probabilities, cis_weights,
                                    trans_weights, cis_dd_power, trans_dd, cis_lengths):     
    cis_trans_mask = cis_trans_mask_slice(cis_lengths, slice_x, slice_y)
    gc_interaction = log_gc_interactions_slice(state_probabilities, cis_weights, trans_weights,
                                               cis_trans_mask, slice_x, slice_y)
    dd_interaction = log_dd_interactions_slice(cis_dd_power, trans_dd, np.sum(cis_lengths),
                                               cis_trans_mask, slice_x, slice_y) 
    matrix =  np.exp(gc_interaction + dd_interaction)
    
    return matrix

class FitMatrix:
    def __init__(self, fit):
        self.fit = fit
        n = np.sum(fit['cis_lengths'])
        self.chr_offsets = np.concatenate([[0], np.cumsum(fit['cis_lengths'])])
        self.shape = (n, n)
    
    def get_chr(self, num, normalize=True):
        index = num - 1
        if index < 0 or num >= self.chr_offsets.size:
            raise IndexError("chr index out of range")
        start = self.chr_offsets[index]
        end = self.chr_offsets[index+1]
        
        chr_slice = self[start:end, start:end]
        if normalize:
            chr_slice /= np.nansum(chr_slice)/2
    
        return chr_slice
    
    def chr_select(self, num):
        if not isiterable(num):
            num = [ num ]
        num = np.array([ n - 1 for n in num ])
        if (num < 0).any() or num >= self.chr_offsets.size:
            raise IndexError("chr index out of range")
            
        new_fit = hic.chr_select(self.fit, num)
        
        return FitMatrix(new_fit)
    
    def __getitem__(self, key):
        try:
            x, y = key
        except:
            raise ValueError("Must use 2-d indexing")
        if not isinstance(x, slice):
            x = slice(x, x+1)
        if not isinstance(y, slice):
            y = slice(y, y+1)
        return generate_interactions_mat_slice(x, y, **self.fit)

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

@curry
def likelihood_ratio(interactions_mat, fit_mat):
    nz = array_utils.get_lower_triangle((interactions_mat != 0) & (fit_mat != 0), k=-1)
    return np.exp(likelihood(interactions_mat, fit_mat, nz) - likelihood(interactions_mat, interactions_mat, nz))

per_chr_fit_name = lambda c, s=2: f"/srv01/technion/hagaik/storage/stretch/per_chr/chr{c}/chr{c}_{s}st_50000_best.npz"
per_chr_fit = lambda c: FitMatrix(gc_datafile.load_params(per_chr_fit_name(c)))
per_chr_resampled_fit_name = lambda c, r, s=2: f"/srv01/technion/hagaik/storage/stretch/per_chr/chr{c}/optimal/chr{c}_{s}st_50000_1.0_run{r}.npy"
per_chr_resampled_fit = lambda c, r: np.load(per_chr_resampled_fit_name(c, r))


def calc_performance_stats_perchr(st, chromosomes):
    pearson_corrs = np.empty((len(chromosomes),))
    spearman_corrs = np.empty((len(chromosomes),))
    lambda_mean_diffs = np.empty((len(chromosomes),))
    lr = np.empty((len(chromosomes),))
    normalized_pearsonr = np.empty((len(chromosomes),))
    normalized_spearmanr = np.empty((len(chromosomes),))
    for i, chrom in enumerate(chromosomes):
        print(f"* Chromosome: {chrom}")
        fitted_mat = per_chr_fit(chrom)[:, :]
        normalized_fit = hic.normalize_distance(fitted_mat)
        input_mat = hic.get_matrix_from_coolfile(mcool_filename, 50000, f'chr{chrom}')
        normalized_input = hic.normalize_distance(input_mat)
        print("** Pearson:", end=' ')
        pearson_corrs[i] = calc_pearsonr(input_mat, fitted_mat)
        print(pearson_corrs[i])
        print("** Spearman:", end=' ')
        spearman_corrs[i] = calc_spearmanr(input_mat, fitted_mat)
        print(spearman_corrs[i])
        print("** Normalized pearson:", end=' ')
        normalized_pearsonr[i] = calc_pearsonr(normalized_input, normalized_fit)
        print(normalized_pearsonr[i])
        print("** Normalized spearman:", end=' ')
        normalized_spearmanr[i] = calc_spearmanr(normalized_input, normalized_fit)
        print(normalized_spearmanr[i])            
        print("** LR:", end=' ')
        lr[i] = likelihood_ratio(input_mat, fitted_mat)
        print(lr[i])
    return dict(pearson_corrs=pearson_corrs, spearman_corrs=spearman_corrs,
               normalized_pearson_corrs=normalized_pearsonr, normalized_spearman_corrs=normalized_spearmanr,
                lr=lr)

def calc_optimal_stats_perchr(st, chromosomes, total_runs):
    pearson_corrs = np.empty((len(chromosomes), total_runs))
    spearman_corrs = np.empty((len(chromosomes), total_runs))
    lr = np.empty((len(chromosomes), total_runs))
    normalized_pearsonr = np.empty((len(chromosomes), total_runs))
    normalized_spearmanr = np.empty((len(chromosomes), total_runs))
    for i, chrom in enumerate(chromosomes):
        print(f"* Chromosome: {chrom}")
        print("** Loading model mat")
        model_mat = per_chr_fit(chrom)[:, :]
        for j in range(total_runs):
            run = j + 1
            print(f"** Loading resampled mat run {run}")
            res_mat = per_chr_resampled_fit(chrom, run)
            print('*** Calculating correlations')
            get_stats = curry(juxt(calc_pearsonr,
                             calc_spearmanr,
                             likelihood_ratio,
                             lambda x, y: calc_pearsonr(hic.normalize_distance(x), hic.normalize_distance(y)),
                             lambda x, y: calc_spearmanr(hic.normalize_distance(x), hic.normalize_distance(y))
                            ))
            stats = get_stats(model_mat, res_mat)
            pearson_corrs[i, j], spearman_corrs[i, j], lr[i, j], normalized_pearsonr[i, j], normalized_spearmanr[i, j] = stats
            print("*** Mean pearson:", pearson_corrs[i, j])
            print("*** Mean spearman:", spearman_corrs[i, j])
            print("*** Mean LR:", lr[i, j])
            print("*** Mean normalized pearson:", normalized_pearsonr[i, j])
            print("*** Mean normalized spearman:", normalized_spearmanr[i, j])

    return dict(pearson_corrs=pearson_corrs, spearman_corrs=spearman_corrs,
                normalized_pearson_corrs=normalized_pearsonr, lr=lr,
                normalized_spearman_corrs=normalized_spearmanr)

if __name__ == '__main__':
    if 0:
        print("Calclating stats")
        perchr_performance_stats = calc_performance_stats_perchr(2, range(1, 23))
        np.savez('/srv01/technion/hagaik/storage/stretch/per_chr/performance_stats.npz', **perchr_performance_stats)
    if 1:
        print("Calclating optimal stats")
        perchr_optimal_stats = calc_optimal_stats_perchr(2, range(1, 23), 20)
        np.savez('/srv01/technion/hagaik/storage/stretch/per_chr/optimal_stats.npz', **perchr_optimal_stats)

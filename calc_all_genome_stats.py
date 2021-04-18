import numpy as np
from scipy import stats
import time

import gc_model as gc
from gc_datafile import load_params
import array_utils
import hic_analysis as hic
from toolz.curried import *

fit_to_mat = lambda fit: gc.generate_interactions_matrix(**fit)

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
    return np.exp(likelihood(interactions_mat, fit_mat) - likelihood(interactions_mat, interactions_mat))

@curry
def calc_pearsonr(mat1, mat2):
    nn = np.isfinite(mat1) & np.isfinite(mat2)
    return stats.pearsonr(mat1[nn], mat2[nn])[0]

@curry
def calc_spearmanr(mat1, mat2):
    nn = np.isfinite(mat1) & np.isfinite(mat2)
    return stats.spearmanr(mat1[nn], mat2[nn])[0]

def log(msg):
    print(time.ctime(), ":", msg)

def calc_stats(all_mat, states, instances=10):
    pearson_array = np.empty((len(states), instances))
    spearman_array = np.empty((len(states), instances))
    lr_array = np.empty((len(states), instances))
    for i, s in enumerate(states):
        log(f"Calculating stats for state {s}")
        for k in range(instances):
            log(f"Starting run {k+1}")
            try:
                fit = load_params("/srv01/technion/hagaik/storage/paper/runs/all_diag_{s}st_100000_run{k+1}.npz")
            except FileNotFoundError:
                print("Not found, skipping")
                pearson_array[i, k] = spearman_array[i, k] = lr_array[i, k] = np.nan
                continue
            fit_mat = fit_to_mat(fit)
            pearson_array[i, k] = calc_pearsonr(all_mat, fit_mat)
            log(f"Pearson: {pearson_array[i, k]}")
            spearman_array[i, k] = calc_spearmanr(all_mat, fit_mat)
            log(f"Spearman: {spearman_array[i, k]}")
            lr_array[i, k] = likelihood_ratio(all_mat, fit_mat)
            log(f"LR: {lr_array[i, k]}")
    return dict(pearson=pearson_array, spearman=spearman_array, lr=lr_array)
            

log("Reading source matrix")
all_mat = hic.preprocess(hic.get_matrix_from_coolfile("/storage/md_kaplan/SharedData/Rao-Cell-2015/cool/Rao2014-GM12878-MboI-allreps-filtered.mcool", 100000, 'all'))
chr_lengths = hic.get_chr_lengths("/storage/md_kaplan/SharedData/Rao-Cell-2015/cool/Rao2014-GM12878-MboI-allreps-filtered.mcool", 100000, ['all'])


st_range = list(range(2, 9))
stats = calc_stats(all_mat, st_range)
output_file = '/srv01/technion/hagaik/storage/paper/runs/all_genome_stats.npy'
log(f"Saving file to {output_file}")
np.savez_compressed(output_file, **stats)

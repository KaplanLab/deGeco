import warnings
import glob
import sys
from scipy import stats
import numpy as np
import sys

import gc_model as gc
import gc_datafile
import array_utils
import hic_analysis as hic
from toolz.curried import *

mcool_filename = mcool_filename = '/srv01/technion/hagaik/storage/Rao_GM12878_zoomified.mcool'
filename_50k_best = lambda d, s: glob.glob(f"/srv01/technion/hagaik/storage/stretch/{d}/all_no_ym_stretched_{s}st_50000_z*_best.npz")[0]
fit_50k_best = curry(compose(gc_datafile.load_params, filename_50k_best))
fit_50k_gm = fit_50k_best("500k_100k_50k_no_ym")
fit_50k_gm_chr = curry(lambda s, c: hic.normalize_distance(gc.generate_interactions_matrix(**hic.chr_select(fit_50k_gm(s), c))))

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

def calc_stats(res, chroms, get_fit_func, st):
    fit_spearman = []
    fit_pearson = []
    optimal_spearman = []
    optimal_pearson = []

    for i in chroms:
        print(f"*** chr{i} ***")
        chr_name = f'chr{i}' if i < 23 else 'chrX'
        data = hic.preprocess(hic.get_matrix_from_coolfile(mcool_filename, res, chr_name))
        normalized = hic.normalize_distance(data)
        
        print("Model")
        recons = get_fit_func(st, [i-1])
        fit_spearman.append(calc_spearmanr(normalized, recons))
        fit_pearson.append(calc_pearsonr(normalized, recons))
        print("Optimal")
        spearman_correlations = []
        pearson_correlations = []
        for j in range(20):
            print(f"* Run {j+1}")
            resampled = np.load(f"/srv01/technion/hagaik/storage/stretch/500k_100k_50k_no_ym/resampled_{st}st/{chr_name}_{st}st_50000_1.0_run{j+1}.npy")
            normalized = hic.normalize_distance(resampled)    
            spearman_correlations.append(calc_spearmanr(normalized, recons))
            pearson_correlations.append(calc_pearsonr(normalized, recons))
        optimal_spearman.append(spearman_correlations)
        optimal_pearson.append(pearson_correlations)
    return fit_spearman, fit_pearson, optimal_spearman, optimal_pearson

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} STATES")
        sys.exit(1)
    st = sys.argv[1]
    filename = f"/srv01/technion/hagaik/storage/stretch/500k_100k_50k_no_ym/stats_{st}st.npz"
    print(f"Calculating stats with states={st}, saving to", filename)
    fit_spearman, fit_pearson, optimal_spearman, optimal_pearson = calc_stats(res=50000, chroms=range(1, 23), get_fit_func=fit_50k_gm_chr, st=st)
    np.savez(filename, **{
            f'fit_spearman{st}': fit_spearman,
            f'fit_pearson{st}': fit_pearson,
            f'optimal_spearman{st}': optimal_spearman,
            f'optimal_pearson{st}': optimal_pearson,
    })

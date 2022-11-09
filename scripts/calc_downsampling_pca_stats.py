import warnings
import glob
import sys
import itertools

from scipy import stats
import numpy as np
from toolz.curried import *
from sklearn.decomposition import PCA

sys.path.append('.')
import gc_model as gc
import gc_datafile
import array_utils
import hic_analysis as hic

batch_load = compose(map(np.load), glob.glob)
downsampled_chr19 = lambda res, pct: hic.get_matrix_from_coolfile(f'/storage/md_kaplan/hagaik/chr19_downsampled/chr19_downsampled_{float(pct)}.mcool', res, 'all')

resolutions = (10000, 20000, 100000, 500000)
sample_rates = (0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1)

@curry
def calc_pearsonr(mat1, mat2):
    nn = np.isfinite(mat1) & np.isfinite(mat2)
    return stats.pearsonr(mat1[nn], mat2[nn])[0]

@curry
def calc_spearmanr(mat1, mat2):
    nn = np.isfinite(mat1) & np.isfinite(mat2)
    return stats.spearmanr(mat1[nn], mat2[nn])[0]

def pca_prep(mat, epsilon=1e-9):
    nn = ~np.isnan(mat).all(axis=1)
    normalized_mat = hic.normalize_distance(mat)
    mat_nn = normalized_mat[nn, :][:, nn]
    np.fill_diagonal(mat_nn, 0)
    mat_nn += epsilon
    gcmat = np.corrcoef(mat_nn)

    return gcmat, nn

def pca_compartments(gcmat):
    return PCA(n_components=1, svd_solver='full').fit_transform(gcmat)[:, 0]

def pca_reconstruction(gc_vec):
    gc_vec_mat = gc_vec[:, None]
    return gc_vec_mat @ gc_vec_mat.T

def expand_mat(mat, nn):
    ret = np.full((nn.size, nn.size), np.nan)
    ret[np.outer(nn, nn)] = mat.flatten()

    return ret

def calc_performance_stats(resolutions, sample_rates):
    gc_pearsonr = np.empty((len(resolutions), len(sample_rates)))
    gc_spearmanr = np.empty((len(resolutions), len(sample_rates)))
    gcdata_pearsonr = np.empty((len(resolutions), len(sample_rates)))
    gcdata_spearmanr = np.empty((len(resolutions), len(sample_rates)))
    mean_diff = np.empty((len(resolutions), len(sample_rates)))
    for i, res in enumerate(resolutions):
        print(f"* Resolution: {res}")
        input_mat = downsampled_chr19(res, 1)
        gcmat_input_small, nn_input = pca_prep(input_mat)
        gcmat_input = expand_mat(gcmat_input_small, nn_input)
        pca_full = gc.expand_by_mask(pca_compartments(gcmat_input_small), nn_input)
        for j, pct in enumerate(sample_rates):
            print(f"** Sample rate: {pct:.2%}")
            print("*** Loading matrices:", end=' ')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                downsampled_input = downsampled_chr19(res, pct)
                gcmat_downs, nn_downs = pca_prep(downsampled_input)
                pca_downs = gc.expand_by_mask(pca_compartments(gcmat_downs), nn_downs)
                recons = pca_reconstruction(pca_downs)
            print("done")

            print("*** gc pearson:", end=' ')
            gc_pearsonr[i, j] = calc_pearsonr(gcmat_input, recons)
            print(gc_pearsonr[i, j])
            print("*** gc spearman:", end=' ')
            gc_spearmanr[i, j] = calc_spearmanr(gcmat_input, recons)
            print(gc_spearmanr[i, j])            
            print("*** gc data pearson:", end=' ')
            gcdata_pearsonr[i, j] = calc_pearsonr(input_mat, recons)
            print(gcdata_pearsonr[i, j])
            print("*** gc data spearman:", end=' ')
            gcdata_spearmanr[i, j] = calc_spearmanr(input_mat, recons)
            print(gcdata_spearmanr[i, j])            
            print("*** vector diff:", end=' ')
            mean_diff[i, j] = np.nanmean(np.abs(pca_downs - pca_full))
            print(mean_diff[i, j])            
    return dict(gc_pearson_corrs=gc_pearsonr, gc_spearman_corrs=gc_spearmanr,
                gcdata_pearson_corrs=gcdata_pearsonr, gcdata_spearman_corrs=gcdata_spearmanr,
                mean_diff = mean_diff
            )

if __name__ == '__main__':
    if 1:
        print(f"Calculating performance stats for PCA")
        performance_stats = calc_performance_stats(resolutions, sample_rates)
        np.savez(f'/srv01/technion/hagaik/storage/chr19_downsampled/2states/pca_stats_vs_allreads.npz', **performance_stats)

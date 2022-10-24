#!/usr/bin/env python
import itertools
import os
import sys
import glob

import numpy as np
import pandas as pd

sys.path.append('.')
import gc_datafile
import hic_analysis as hic

### Global variables used throughout the analysis
# location of scHi-C data, downloaded from https://noble.gs.washington.edu/proj/schic-topic-model/.
# Should contain subdirs that include both .matrix.gz and .labeled files.
SC_DATA_DIR = "sc_data/"
CELL_LINE = 'GM12878' # cell line from experiments to use

# Location of bulk Hi-C data
MCOOL_FILENAME_HG19 = 'hic/Rao2014-GM12878-MboI-hg19.mcool' # scHi-C data is in hg19

# Location of all-genome model fit
FIT_7ST_FILENAME = "output/all_no_ym_7st_100000_best.npz"

CHR_NUM = 19
READ_COUNT_THRESHOLD = 5 # Use only cells with at least this many reads in the mixed region
MARGIN = 4 # How much bins to remove around the main diagonal of each row

### Functions
def sciHiC2mat(filename, size=None):
    """
    Read a .matrix.gz file and return a dense numpy matrix.

    If size is given, the matrix will be padded with zeros to dimensions size x size.
    """
    t = pd.read_csv(filename + '.gz', delimiter='\t', header=None, names=['bin1_id', 'bin2_id', 'count', 'balanced', 'chr1_name', 'chr2_name'])
    if size:
        mat_size = size
    else:
        mat_size = np.maximum(t['bin1_id'].max(), t['bin2_id'].max()) + 1
    mat = np.full((mat_size, mat_size), 0, dtype=float)
    for i, r in t.iterrows():
        i = r['bin1_id']
        j = r['bin2_id']
        v = r['count']
        mat[i, j] = mat[j, i] = v
    return mat

def read_sc_dir(dir_pattern: str, cell_line: str=None, **kwargs):
    """
    Read all scHi-C data files (of optional cell_line) from all directories matching dir_pattern.

    Extra args are passed to sciHiC2mat.
    """
    unified_filenames = []

    for dirname in glob.glob(dir_pattern):
        dirname = os.path.normpath(dirname)
        labeled_name = os.path.join(dirname, os.path.basename(dirname) + '.labeled')
        labeled = pd.read_csv(labeled_name, sep='\t', header=None, names=['filename', 'line'])
        if cell_line:
            filenames = labeled.query('line == @cell_line')['filename']
        else:
            filenames = labeled['filename']
        unified_filenames.append(f'{dirname}/' + filenames)
    unified_filenames = pd.concat(unified_filenames)
    return (sciHiC2mat(f, **kwargs) for f in unified_filenames)

def get_row_groups(bulk):
    """
    Use the 7-state fit to group chr19 rows into rows belonging to state1, state2 and the mixed state. Also return the
    chromosome after removing the main diagonal from these groups (i.e. all bins besides the s1,s2,mixed bins)
    """
    fit_7st_100k = gc_datafile.load_params(FIT_7ST_FILENAME)
    fit_7st_100k_chr19 = hic.chr_select(fit_7st_100k, [18])
    mixed_bins =  (fit_7st_100k_chr19['state_probabilities'][:, 0] > 0.5) \
                & (fit_7st_100k_chr19['state_probabilities'][:, 6] > 0.1)
    s1_bins =     (fit_7st_100k_chr19['state_probabilities'][:, 0] > 0.4) \
                & (fit_7st_100k_chr19['state_probabilities'][:, 6] < 0.1)  \
                & (fit_7st_100k_chr19['state_probabilities'][:, 2] > 0.1)\
                & (np.arange(fit_7st_100k_chr19['state_probabilities'].shape[0]) > 190) # Remove rows whose diaongal is near the mixed zone
    s2_bins = (fit_7st_100k_chr19['state_probabilities'][:, 0] < 0.3) & \
              (fit_7st_100k_chr19['state_probabilities'][:, 6] > 0.2) & \
              (fit_7st_100k_chr19['state_probabilities'][:, 2] < 0.3) & \
              (fit_7st_100k_chr19['state_probabilities'][:, 1] < 0.3)

    mixed_bins_from_100k = np.unique(np.where(mixed_bins)[0] //5)
    mixed_bins_from_100k = np.delete(mixed_bins_from_100k, [0,6]) # Remove bad row
    mixed_bins_from_100k = mixed_bins_from_100k[~np.isnan(bulk[mixed_bins_from_100k]).all(1)]
    s1_bins_from_100k = np.unique(np.where(s1_bins)[0] //5)
    s1_bins_from_100k = np.delete(s1_bins_from_100k, [0,2,3]) # Remove bad row
    s2_bins_from_100k = np.unique(np.where(s2_bins)[0] //5)
    s2_bins_from_100k = np.delete(s2_bins_from_100k, [10, 8, 1])[4:-5] # Remove bad rows
    diagonal_500k = np.concatenate((mixed_bins_from_100k, s1_bins_from_100k, s2_bins_from_100k))

    return s1_bins_from_100k, s2_bins_from_100k, mixed_bins_from_100k

def bulk_to_probabilities(bulk_bins, epsilon=1e-10):
    """ Convert slice of bulk Hi-C matrix into a probability vector """
    b = np.sum(bulk_bins, axis=0)
    b /= b.sum() # Normalize to probabilities
    b += epsilon # Avoid zero-probability cells
    b /= b.sum()

    return b

def profile_log_likelihood(probabilities, reads):
    """ Calculate the log-likelihood of the given reads from some probability profile """
    log_factorial = lambda a: np.sum(np.log(1 + np.arange(a)))
    log_factorial_batch = lambda a: np.apply_along_axis(log_factorial, 1, a[:, None])

    log_coeff = log_factorial(reads.sum())- np.sum(log_factorial_batch(reads))
    log_p = np.sum(reads * np.log(probabilities))

    return log_coeff + log_p

def sc_read_counts(sc_data_dir, cell_line):
    """
    Return a vector of read counts in mixed bins for each cell in the experiment.
    """
    global matrix_size
    global chr_slice
    global mixed_bins
    global nn

    reads = []
    for i, mat in enumerate(read_sc_dir(f"{sc_data_dir}/*.R?/", cell_line, size=matrix_size)):
        print(f"cell number: {i+1}", end='\r', flush=True)
        mat_chr = mat[chr_slice, chr_slice]
        reads.append(np.sum(slice_and_ignore_diag(mat_chr, mixed_bins, margin=MARGIN)[:, nn]))
    print()
    reads = np.array(reads)
    return reads

def simulate(nreads, profiles, profile_probs, rng):
    """
    Simulate a scHi-C read vector on a population len(nreads) cells that consists of len(profiles) subpopoulations,
    and profile_probs is the fraction of each subpopulation.

    profiles should contain a list of read probability vectors. Each cell i will be assigned a profile randomly using
    profile_probs and then nreads[i] reads will be sampled from the chosen probability profile.

    A matrix of reads of size cells x profile_length will be returned.
    """
    if np.ndim(profile_probs) == 0:
        profile_probs = np.array([profile_probs])
    if len(profile_probs) != len(profiles):
        if len(profile_probs) + 1 != len(profiles):
            raise ValueError("profile_probs must be same length or one less than profiles")
        profile_probs = np.append(profile_probs, [1-np.sum(profile_probs)])
    else:
        profile_probs /= np.sum(profile_probs)
    assert np.all(profile_probs >= 0)

    nprofiles = np.shape(profiles)[0]
    sampled_profiles = np.empty((np.shape(nreads)[0], np.shape(profiles)[1]))
    for i, r in enumerate(nreads):
        profile_idx = rng.choice(nprofiles, p=profile_probs)
        #print(profile_idx)
        profile = profiles[profile_idx]
        sampled_profiles[i] = rng.multinomial(r, profile)

    return sampled_profiles

def calc_score(profiles, cell_reads):
    """
    Calculate loglikelihood score for each simulated cell against the given probability profile
    """
    all_profiles = np.concatenate((profiles['population-level'], profiles['cell-level']))
    scores = np.empty((cell_reads.shape[0], all_profiles.shape[0]))
    for i, r in enumerate(cell_reads):
        for j, p in enumerate(all_profiles):
            scores[i, j] = profile_log_likelihood(p, r)
    return scores

def slice_and_ignore_diag(mat, indices, margin):
    """
    Return mat[indices] and zero `margin` columns around the diagonal in each row.
    """
    sliced = mat[indices]
    for row, i in zip(sliced, indices):
        row[i] = 0
        lmargin = np.maximum(i-margin, 0)
        rmargin = np.minimum(i+1+margin, row.size)
        row[lmargin:i] = 0
        row[i+1:rmargin] = 0
    return sliced

def run_experiment(profiles, reads):
    """
    Run simulation of population- and cell-level mix and calculate likelihoods for simulations and data
    """
    global matrix_size
    global chr_slice
    global mixed_bins
    global nn

    rng = np.random.default_rng(seed=1)
    print("experiment: simulating population-level")
    simulation_pl = simulate(reads, profiles['population-level'], 0.5, rng)
    scores_pl = calc_score(profiles, simulation_pl)
    lr_pl = scores_pl[:, 2] - np.max(scores_pl[:, :2], axis=1)

    rng = np.random.default_rng(seed=1)
    print("experiment: simulating cell-level")
    simulation_cl = simulate(reads, profiles['cell-level'], 0.5, rng)
    scores_cl = calc_score(profiles, simulation_cl)
    lr_cl = scores_cl[:, 2] - np.max(scores_cl[:, :2], axis=1)

    likelihoods = []
    for i, mat in enumerate(read_sc_dir(f"{SC_DATA_DIR}/*.R?/", CELL_LINE, size=matrix_size)):
        print(f"experiment: calculating score for data. cell number: {i+1}/{len(reads)}", end='\r', flush=True)
        mat_chr19 = mat[chr_slice, chr_slice]
        mat_mixed = slice_and_ignore_diag(mat_chr19, mixed_bins, margin=MARGIN)[:, nn].sum(axis=0)
        res = calc_score(profiles, np.array([mat_mixed]))[0]
        likelihoods.append(res)
    print()
    likelihoods = np.array(likelihoods)
    lr_data = likelihoods[:, 2] - np.max(likelihoods[:, :2], axis=1)

    return dict(pl=dict(simulation=simulation_pl, scores=scores_pl, lr=lr_pl),
                cl=dict(simulation=simulation_cl, scores=scores_cl, lr=lr_cl),
                data=dict(scores=likelihoods, lr=lr_data))


#### START ####
# Information derived from Hi-C matrix and fit
print("Reading bulk Hi-C data")
bulk = hic.get_matrix_from_coolfile(MCOOL_FILENAME_HG19, 500000, 'chr19')
nn = ~np.isnan(bulk[0]) # mask of non-nan columns
bulk_normalized = hic.normalize_distance(bulk)
chr_offsets = np.cumsum(np.concatenate([(0,), hic.get_chr_lengths(MCOOL_FILENAME_HG19, 500000, ['all'])]))
matrix_size = chr_offsets[-1]
chr_slice = slice(chr_offsets[CHR_NUM-1], chr_offsets[CHR_NUM])
s1_bins, s2_bins, mixed_bins = get_row_groups(bulk)

reads_filename = f"{SC_DATA_DIR}/{CELL_LINE}_reads_mix.npy"
if os.path.exists(reads_filename):
    print("Using cached read counts from", reads_filename)
    reads = np.load(reads_filename)
else:
    print("Calculating read counts")
    reads = sc_read_counts(SC_DATA_DIR, CELL_LINE)
    print("Caching read counts to", reads_filename)
    np.save(reads_filename, reads)

reads_mask = reads >= READ_COUNT_THRESHOLD

# Simulation profiles
profiles = dict()
bulk_dd = bulk / bulk_normalized
profiles['ddonly'] = {
    'population-level': [
        bulk_to_probabilities(slice_and_ignore_diag(bulk_dd, s1_bins, margin=MARGIN)[:, nn]),
        bulk_to_probabilities(slice_and_ignore_diag(bulk_dd, s2_bins, margin=MARGIN)[:, nn])
    ],
    'cell-level': [
        bulk_to_probabilities(slice_and_ignore_diag(bulk_dd, mixed_bins, margin=MARGIN)[:, nn])
    ]
}

profiles['withgc'] = {
    'population-level': [
        bulk_to_probabilities(slice_and_ignore_diag(bulk, s1_bins, margin=MARGIN)[:, nn]),
        bulk_to_probabilities(slice_and_ignore_diag(bulk, s2_bins, margin=MARGIN)[:, nn])
    ],
    'cell-level': [
        bulk_to_probabilities(slice_and_ignore_diag(bulk, mixed_bins, margin=MARGIN)[:, nn])
    ]
}


likelihoods_ddonly_filename = f"{SC_DATA_DIR}/likelihoods_ddonly.npy"
if os.path.exists(likelihoods_ddonly_filename):
    print("Using cached ddonly experiment results from", likelihoods_ddonly_filename)
    ddonly_experiment = np.load(likelihoods_ddonly_filename, allow_pickle=True)[()]
else:
    print("Running ddonly experiment")
    ddonly_experiment = run_experiment(profiles['ddonly'], reads)
    print("Caching ddonly experiment results to", likelihoods_ddonly_filename)
    np.save(likelihoods_ddonly_filename, ddonly_experiment)

likelihoods_withgc_filename = f"{SC_DATA_DIR}/likelihoods_withgc.npy"
if os.path.exists(likelihoods_withgc_filename):
    print("Using cached withgc experiment results from", likelihoods_withgc_filename)
    withgc_experiment = np.load(likelihoods_withgc_filename, allow_pickle=True)[()]
else:
    print("Running withgc experiment")
    withgc_experiment = run_experiment(profiles['withgc'], reads)
    print("Caching withgc experiment results to", likelihoods_withgc_filename)
    np.save(likelihoods_withgc_filename, withgc_experiment)

print("Calculating mixed log ratio")
mixed_log_ratio = dict()
mixed_log_ratio['population-level'] = withgc_experiment['pl']['lr'] - ddonly_experiment['pl']['lr']
mixed_log_ratio['cell-level'] = withgc_experiment['cl']['lr'] - ddonly_experiment['cl']['lr']
mixed_log_ratio['data'] = withgc_experiment['data']['lr'] - ddonly_experiment['data']['lr']

filename = f"{SC_DATA_DIR}/mixed_log_ratio.npz"
np.savez(filename, **mixed_log_ratio, reads_mask=reads_mask)
print("Saved MLR in:", filename)

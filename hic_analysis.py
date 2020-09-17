import numpy as np
import warnings

from matplotlib import pyplot as plt
import cooler

from array_utils import ensure_symmetric, normalize_tri_l1, remove_main_diag

def get_chr_lengths(mcool_filename, experiment_resolution, chromosomes):
    """
    Return a tuple of chromosome lengths, in bins of the given resolution, based on the given mcool file.

    :param str mcool_filename: The file to read from
    :param int experiment_resolution: The experiment resolution (bin size) to read
    :param iterable chromosomes: The chromosomes to return. Format should be 'chrX'

    :return: tuple of ints: Lengths of requested chromosomes, in bins
    """
    coolfile = f'{mcool_filename}::/resolutions/{experiment_resolution}'
    c = cooler.Cooler(coolfile)

    ranges = ( c.extent(chrom) for chrom in chromosomes )
    lengths = ( end - start for start, end in ranges )
    
    return tuple(lengths)

def get_matrix_from_coolfile(mcool_filename, experiment_resolution, chromosome1, chromosome2=None):
    """
    Return a numpy matrix (balanced Hi-C) from an mcool file.

    :param str mcool_filename: The file to read from
    :param int experiment_resolution: The experiment resolution (bin size) to read
    :param str chromosome1: The chromosome to look for. Format should be: chrXX
    :param str chromosome2: Second chromosome to look for. If specified, both chrs will be concatenated,
                            including trans regions.
    :return: array: a numpy matrix containing the data of the requested chromosomes at the requested resolution.
    """
    coolfile = f'{mcool_filename}::/resolutions/{experiment_resolution}'
    c = cooler.Cooler(coolfile)

    if chromosome2 is None:
        (start_idx, end_idx) = c.extent(chromosome1)
        experimented_cis_interactions = c.matrix()[start_idx:end_idx,start_idx:end_idx]

        return experimented_cis_interactions

    (start_cis1, end_cis1) = c.extent(chromosome1)
    (start_cis2, end_cis2) = c.extent(chromosome2)

    mat = c.matrix()
    cis_interactions1 = mat[start_cis1:end_cis1,start_cis1:end_cis1]
    cis_interactions2 = mat[start_cis2:end_cis2,start_cis2:end_cis2]
    trans_interactions1 = mat[start_cis1:end_cis1, start_cis2:end_cis2]
    trans_interactions2 = mat[start_cis2:end_cis2, start_cis1:end_cis1]

    alll_interactions = np.vstack([
        np.hstack((cis_interactions1, trans_interactions1)),
        np.hstack((trans_interactions2, cis_interactions2))
        ])

    return alll_interactions

def get_selector_from_coolfile(mcool_filename, resolution):
    """
    Return a Cooler selector from an mcool file. A selector can be sliced to get the wanted parts of the matrix,
    without loading it all into memory first.

    :param str mcool_filename: The file to read from
    :param int experiment_resolution: The experiment resolution (bin size) to read
    :return: A cooler.api.RangeSelector2D object
    """
    coolfile = f'{mcool_filename}::/resolutions/{resolution}'
    c = cooler.Cooler(coolfile)

    return c.matrix()

def remove_unusable_bins(interactions):
    """
    Remove all bins that have NaN in all of their values
    """
    non_nan_mask = ~np.isnan(interactions).all(1)

    return interactions[:, non_nan_mask][non_nan_mask, :]

def preprocess(interactions):
    """
    Perform various arrangements on the input matrix to make it easier to analyze. Removes the main diagonal, 
    ensures matrix is symmetric and normalizes the L1 norm.

    :param array interactions_mat: Interactions matrix
    :return: Interactions matrix after processing
    :rtype: array
    """
    symmetric = ensure_symmetric(interactions)
    no_main_diag = remove_main_diag(symmetric)
    l1_normalized = normalize_tri_l1(no_main_diag)

    return l1_normalized

def normalize_distance(interactions):
    """
    Normalize interactions matrix to remove the distance effects from interactions.
    This is done by ensuring all diagonals have the same sum.

    :param array interactions: Hi-C interactions matrix
    :return: distance-normalized interaction matrix
    :rtype: 2D array
    """
    number_of_bins = interactions.shape[0]
    normalized = np.full_like(interactions, np.nan)
    for diag_num in range(1, number_of_bins):
        diag = interactions.diagonal(diag_num)
        if np.isnan(diag).all():
            continue
        diag_mean = np.nanmean(diag)
        if diag_mean == 0:
            normalized_diag = 0
        else:
            normalized_diag = diag / diag_mean
        np.fill_diagonal(normalized[:, diag_num:], normalized_diag) # upper diag
        np.fill_diagonal(normalized[diag_num:], normalized_diag) # lower diag

    # Ensure total sum of each half is 1
    normalized /= np.nansum(np.tril(normalized, -1))

    return normalized

def zeros_to_nan(data):
    no_zeros = np.full_like(data, np.nan)
    mask = data != 0
    no_zeros[mask] = data[mask]

    return no_zeros

def safe_div(a, b):
    return zeros_to_nan(a) / zeros_to_nan(b)

def safe_log(a):
    return np.log(zeros_to_nan(a))

def remove_nan(data):
    nan_idx = np.isnan(data)

    return data[~nan_idx]

def merge_by_diagonal(a, b):
    return remove_main_diag(np.tril(a, k=-1) + np.triu(b, k=1))

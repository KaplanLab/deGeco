import numpy as np
from matplotlib import pyplot as plt
import cooler

def get_matrix_from_coolfile(mcool_filename, experiment_resolution, chromosome):
    """
    Return a numpy matrix (balanced Hi-C) from an mcool file.

    :param str mcool_filename: The file to read from
    :param int experiment_resolution: The experiment resolution (bin size) to read
    :param str chromosome: The chromosome to look for. Format should be: chrXX
    :return: A numpy matrix containing the data of the requested chromosome at the requested resolution
    """
    coolfile = f'{mcool_filename}::/resolutions/{experiment_resolution}'
    c = cooler.Cooler(coolfile)

    (start_idx, end_idx) = c.extent(chromosome)
    experimented_cis_interactions = c.matrix()[start_idx:end_idx,start_idx:end_idx]

    return experimented_cis_interactions

def normalize_tri_l1(interactions):
    """
    Divides the input matrix by the L1 norm of the lower triangle
    """
    return interactions / np.nansum(np.tril(interactions, -1))

def ensure_symmetric(interactions):
    """
    Make A[i, j] = A[j, i] = max(A[i, j], A[j, i])
    """
    return np.maximum(interactions, interactions.T)
    
def remove_main_diag(interactions):
    """
    Put NaNs in the main diagonal so it's removed from all calculations.
    """
    nan_in_diag = interactions.copy()
    np.fill_diagonal(nan_in_diag, np.nan)

    return nan_in_diag

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
        diag_sum = np.nansum(diag)
        if diag_sum == 0:
            continue
        normalized_diag = diag / diag_sum
        np.fill_diagonal(normalized[:, diag_num:], normalized_diag) # upper diag
        np.fill_diagonal(normalized[diag_num:], normalized_diag) # lower diag

    # Ensure total sum of each half is 1
    normalized /= np.nansum(np.tril(normalized, -1))

    return normalized

def zeros_to_nan(data):
    no_zeros = data.copy()
    no_zeros[data == 0] = np.nan

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

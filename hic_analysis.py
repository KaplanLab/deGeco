import numpy as np
import warnings

from matplotlib import pyplot as plt
import cooler

from array_utils import ensure_symmetric, normalize_tri_l1, remove_main_diag
from balance_counts import balance_counts

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

    if chromosomes == ['all'] or chromosomes == 'all':
       chromosomes = c.chromnames
    ranges = ( c.extent(chrom) for chrom in chromosomes )
    lengths = ( end - start for start, end in ranges )
    
    return tuple(lengths)

def get_matrix_from_coolfile(mcool_filename, experiment_resolution, chromosome1, chromosome2=None, **matrix_args):
    """
    Return a numpy matrix (balanced Hi-C) from an mcool file.

    :param str mcool_filename: The file to read from
    :param int experiment_resolution: The experiment resolution (bin size) to read. If None, assume cool file with one res.
    :param str chromosome1: The chromosome to look for. Format should be: chrXX
    :param str chromosome2: Second chromosome to look for. If specified, both chrs will be concatenated,
                            including trans regions.
    :param matrix_args: Other arguments to pass to Cooler.matrix()
    :return: array: a numpy matrix containing the data of the requested chromosomes at the requested resolution.
    """
    if experiment_resolution is not None:
        coolfile = f'{mcool_filename}::/resolutions/{experiment_resolution}'
    else:
        coolfile = f'{mcool_filename}'
    c = cooler.Cooler(coolfile)

    if chromosome1 == 'all':
        return c.matrix()[:, :]

    if chromosome2 is None:
        (start_idx, end_idx) = c.extent(chromosome1)
        experimented_cis_interactions = c.matrix(**matrix_args)[start_idx:end_idx,start_idx:end_idx]

        return experimented_cis_interactions

    (start_cis1, end_cis1) = c.extent(chromosome1)
    (start_cis2, end_cis2) = c.extent(chromosome2)

    mat = c.matrix(**matrix_args)
    cis_interactions1 = mat[start_cis1:end_cis1,start_cis1:end_cis1]
    cis_interactions2 = mat[start_cis2:end_cis2,start_cis2:end_cis2]
    trans_interactions1 = mat[start_cis1:end_cis1, start_cis2:end_cis2]
    trans_interactions2 = mat[start_cis2:end_cis2, start_cis1:end_cis1]

    alll_interactions = np.vstack([
        np.hstack((cis_interactions1, trans_interactions1)),
        np.hstack((trans_interactions2, cis_interactions2))
        ])

    return alll_interactions

def get_sparse_matrix_from_coolfile(mcool_filename, resolution, chromosome1, chromosome2=None, **matrix_args):
    if resolution is not None:
        coolfile = f'{mcool_filename}::/resolutions/{resolution}'
    else:
        coolfile = f'{mcool_filename}'
    c = cooler.Cooler(coolfile)

    if chromosome1 == 'all':
        # If we fetch the entire dataset as sparse values, it's faster this way
        mat = c.pixels(as_dict=True)[:]
        mat['bin1_id'] = mat['bin1_id'].astype(np.int32)
        mat['bin2_id'] = mat['bin2_id'].astype(np.int32)
        balance_weights = c.bins()['weight'][:].to_numpy()
        if matrix_args.get('balance', True):
            mat['count'] = balance_counts(mat['bin1_id'], mat['bin2_id'], mat['count'], balance_weights)

        mat['non_nan_mask'] = ~np.isnan(balance_weights)
        return mat

    if chromosome2 is None:
        start, end = c.extent(chromosome1)
        non_nan_mask = ~np.isnan(c.bins()[start:end]['weight'].to_numpy())
        m = c.matrix(as_pixels=True)[start:end, start:end]
        m[['bin1_id', 'bin2_id']] -= start
    else:
        start1, end1 = c.extent(chromosome1)
        start2, end2 = c.extent(chromosome2)
        if start1 > start2:
            start1, start2 = start2, start1
            end1, end2 = end2, end1
        len1 = end1 - start1

        non_nan_mask1 = ~np.isnan(c.bins()[start1:end1]['weight'].to_numpy())
        m1_1 = c.matrix(as_pixels=True)[start1:end1, start1:end1]
        m1_1[['bin1_id', 'bin2_id']] -= start1

        non_nan_mask2 = ~np.isnan(c.bins()[start2:end2]['weight'].to_numpy())
        m2_2 = c.matrix(as_pixels=True)[start2:end2, start2:end2]
        m2_2[['bin1_id', 'bin2_id']] += -start2 + len1

        m1_2 = c.matrix(as_pixels=True)[start1:end1, start2:end2]
        m1_2['bin1_id'] -= start1
        m1_2['bin2_id'] += -start2 + len1

        non_nan_mask = np.concatenate((non_nan_mask1, non_nan_mask2))
        m = m1_1.append(m1_2).append(m2_2).sort_values(['bin1_id', 'bin2_id'])

    return dict(bin1_id=m['bin1_id'].to_numpy().astype(np.int32), bin2_id=m['bin2_id'].to_numpy().astype(np.int32),
                count=m['balanced'].to_numpy(), non_nan_mask=non_nan_mask)


def preprocess_sprase(sparse_data, dups='fix'):
    bin1_id, bin2_id, count = sparse_data['bin1_id'], sparse_data['bin2_id'], sparse_data['count']
    #_, unique_idx, inverse_idx, unique_counts = np.unique(np.vstack([bin1_id, bin2_id]), axis=1, return_index=True, return_inverse=True, return_counts=True)
    #if dups == 'fix':
    #    bin1_id = bin1_id[unique_idx]
    #    bin2_id = bin2_id[unique_idx]
    #    counts_uniq = np.zeros(unique_idx.size)
    #    for c, t in zip(count, inverse_idx):
    #        counts_uniq[t] += c
    #    count = counts_uniq

    #elif dups == 'ignore':
    #    bin1_id = bin1_id[unique_idx]
    #    bin2_id = bin2_id[unique_idx]
    #    count = counts_uniq
    #elif dups == 'remove':
    #    idx_mask = unique_counts == 1
    #    bin1_id = bin1_id[unique_idx[idx_mask]]
    #    bin2_id = bin2_id[unique_idx[idx_mask]]
    #    count = count[unique_idx[idx_mask]]
    #else:
    #    raise ValueError(f"Invalid dup: {dups}")

    count[bin1_id == bin2_id] = np.nan

    return dict(bin1_id=bin1_id, bin2_id=bin2_id, count=count, non_nan_mask=sparse_data.get('non_nan_mask'))

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
    Perform various arrangements on the input matrix to make it easier to analyze. Removes the main diagonal
    and ensures matrix is symmetric.

    :param array interactions_mat: Interactions matrix
    :return: Interactions matrix after processing
    :rtype: array
    """
    symmetric = ensure_symmetric(interactions)
    no_main_diag = remove_main_diag(symmetric)

    return no_main_diag

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

def safe_log(a, epsilon=None):
    """
    log(a + epsilon), so that zeros aren't converted to NaNs. If not given, epsilon = (smallest non-zero value of a)/10
    """
    if epsilon is None:
        epsilon = np.nanmin(a[a!=0]) / 10
    return np.log(a + epsilon)

def remove_nan(data):
    nan_idx = np.isnan(data)

    return data[~nan_idx]

def merge_by_diagonal(a, b):
    return remove_main_diag(np.tril(a, k=-1) + np.triu(b, k=1))

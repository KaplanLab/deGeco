import numpy as np
import warnings
import itertools

import cooler
from pandas import concat

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
    if chromosomes == ['all_no_ym'] or chromosomes == 'all_no_ym':
       chromosomes = [ ch for ch in c.chromnames if ch not in ['chrY', 'chrM'] ]
    ranges = ( c.extent(chrom) for chrom in chromosomes )
    lengths = ( end - start for start, end in ranges )
    
    return tuple(lengths)

def _get_start_chrYM(cooler_obj):
    end = None
    for c in ('chrY', 'chrM'):
        if c in cooler_obj.chromnames:
            start, _ = cooler_obj.extent(c)
            if end is None or start < end:
                end = start
    return end

def get_nn_from_mcool(mcool_filename, resolution, *chromosomes, weight_column='weight'):
    """
    Return a list of masks for each chromosome, marking which bin is usable after balancing.

    :param str mcool_filename: The file to read from
    :param int resolution: The  resolution (bin size) to read
    :param iterable chromosomes: The chromosomes to work on.
    :param str weight_column: The cooler file column that stores the balancing weights

    :return: list of arrays: non-nan masks
    """
    c = cooler.Cooler(f'{mcool_filename}::/resolutions/{resolution}')
    if chromosomes[0] == 'all':
        chromnames = c.chromnames
    elif chromosomes[0] == 'all_no_ym':
        chromnames = [ ch for ch in c.chromnames if ch not in ['chrY', 'chrM', 'Y', 'M'] ]
    else:
        chromnames = list(chromosomes)

    def chr_nn(chromosome):
        chr_start, chr_end = c.extent(chromosome)
        weights = c.bins()[chr_start:chr_end][weight_column]
        return ~np.isnan(weights.to_numpy())

    return [ chr_nn(ch) for ch in chromnames ]

def _read_square_up_to_bin(cooler_obj, end_bin):
    """
    Read the entire hi-c matrix in sparse form up to bin end_bin. This is a more
    memory efficient way of doing c.matrix(as_pixels=True)[:end_bin, :end_bin]
    """
    with cooler.util.open_hdf5(cooler_obj.store, **cooler_obj.open_kws) as h5:
        grp = h5[cooler_obj.root]
        edges = grp["indexes"]["bin1_offset"][:end_bin+1]
        end_pixel_overestimate = edges[-1]
        count = grp['pixels']['count']
        mat = {
            'bin1_id': np.empty(end_pixel_overestimate, dtype=np.int32),
            'bin2_id': np.empty(end_pixel_overestimate, dtype=np.int32),
            'count': np.empty(end_pixel_overestimate, dtype=np.float64),
        }
        start_pos = 0
        for cur_bin, start_pixel, end_pixel in zip(range(end_bin), edges[:-1], edges[1:]):
            cols = grp['pixels']['bin2_id'][start_pixel:end_pixel] 
            mask = cols < end_bin
            end_pos = start_pos + np.sum(mask)
            mat['bin1_id'][start_pos:end_pos] = cur_bin
            mat['bin2_id'][start_pos:end_pos] = cols[mask]
            mat['count'][start_pos:end_pos] = count[start_pixel:end_pixel][mask]
            start_pos = end_pos
        mat['bin1_id'] = mat['bin1_id'][:end_pos]
        mat['bin2_id'] = mat['bin2_id'][:end_pos]
        mat['count'] = mat['count'][:end_pos]

        return mat

def get_matrix_from_coolfile(mcool_filename, experiment_resolution, chromosome1, *chroms, **matrix_args):
    """
    Return a numpy matrix (balanced Hi-C) from an mcool file.

    :param str mcool_filename: The file to read from
    :param int experiment_resolution: The experiment resolution (bin size) to read. If None, assume cool file with one res.
    :param str chromosome1: The chromosome to look for. Format should be: chrXX
    :param list chroms: More chromosome to look for. If specified, they will be concatenated to chromosome1, including
                            trans regions.
    :param matrix_args: Other arguments to pass to Cooler.matrix()
    :return: array: a numpy matrix containing the data of the requested chromosomes at the requested resolution.
    """
    if experiment_resolution is not None:
        coolfile = f'{mcool_filename}::/resolutions/{experiment_resolution}'
    else:
        coolfile = f'{mcool_filename}'
    c = cooler.Cooler(coolfile)
    mat = c.matrix(**matrix_args)

    if chromosome1 == 'all':
        return mat[:, :]

    if chromosome1 == 'all_no_ym':
        end = _get_start_chrYM(c)
        return mat[:end, :end]

    if len(chroms) == 0:
        slice_cis1 = slice(*c.extent(chromosome1))
        cis_interactions1 = mat[slice_cis1, slice_cis1]
        return cis_interactions1

    chroms = (chromosome1,) + chroms

    total_length = np.sum(get_chr_lengths(mcool_filename, experiment_resolution, chroms))
    all_interactions = np.empty((total_length, total_length))
    distances = [0] + [ c.extent(c2)[0] - c.extent(c1)[0] for c1, c2 in zip(chroms[:-1], chroms[1:]) ]
    offsets = np.cumsum(distances)
    for i1, i2 in itertools.combinations_with_replacement(range(len(chroms)), 2):
        c1 = chroms[i1]
        c2 = chroms[i2]
        start1, end1 = c.extent(c1)
        start2, end2 = c.extent(c2)
        len1 = end1 - start1
        len2 = end2 - start2
        all_interactions[offsets[i1]:offsets[i1]+len1, offsets[i2]:offsets[i2]+len2] = mat[start1:end1, start2:end2]
        if i1 != i2:
            all_interactions[offsets[i2]:offsets[i2]+len2, offsets[i1]:offsets[i1]+len1] = mat[start2:end2, start1:end1]

    return all_interactions

def get_sparse_matrix_from_coolfile(mcool_filename, resolution, chromosome1, *chroms, transonly=False, **matrix_args):
    if resolution is not None:
        coolfile = f'{mcool_filename}::/resolutions/{resolution}'
    else:
        coolfile = f'{mcool_filename}'
    c = cooler.Cooler(coolfile)
    weight_column = matrix_args.get('balance', 'weight')

    if chromosome1 == 'all':
        # If we fetch the entire dataset as sparse values, it's faster this way
        mat = c.pixels(as_dict=True)[:]
        mat['bin1_id'] = mat['bin1_id'].astype(np.int32)
        mat['bin2_id'] = mat['bin2_id'].astype(np.int32)
        balance_weights = c.bins()[weight_column][:].to_numpy()
        if matrix_args.get('balance', True):
            mat['count'] = balance_counts(mat['bin1_id'], mat['bin2_id'], mat['count'], balance_weights)

        mat['non_nan_mask'] = ~np.isnan(balance_weights)
        return mat

    if chromosome1 == 'all_no_ym':
        end = _get_start_chrYM(c)
        mat = _read_square_up_to_bin(c, end)
        balance_weights = c.bins()[weight_column][:end].to_numpy()
        if matrix_args.get('balance', True):
            mat['count'] = balance_counts(mat['bin1_id'], mat['bin2_id'], mat['count'], balance_weights)

        mat['non_nan_mask'] = ~np.isnan(balance_weights)
        return mat

    if len(chroms) == 0:
        start, end = c.extent(chromosome1)
        non_nan_mask = ~np.isnan(c.bins()[start:end][weight_column].to_numpy())
        m = c.matrix(as_pixels=True, **matrix_args)[start:end, start:end]
        m[['bin1_id', 'bin2_id']] -= start
    else:
        chroms = (chromosome1,) + chroms
        non_nan_masks = []
        dfs = []
        sizes = [0] + [ c.extent(c1)[1] - c.extent(c1)[0] for c1 in chroms[:-1] ]
        offsets = np.cumsum(sizes)
        for i1, i2 in itertools.combinations_with_replacement(range(len(chroms)), 2):
            c1 = chroms[i1]
            c2 = chroms[i2]
            start1, end1 = c.extent(c1)
            start2, end2 = c.extent(c2)

            if c1 == c2:
                non_nan_masks.append(~np.isnan(c.bins()[start1:end1][weight_column].to_numpy()))
                if transonly:
                    continue

            df = c.matrix(as_pixels=True, **matrix_args)[start1:end1, start2:end2]
            df[['bin1_id']] += -start1 + offsets[i1]
            df[['bin2_id']] += -start2 + offsets[i2]

            dfs.append(df)

        non_nan_mask = np.concatenate(non_nan_masks)
        m = concat(dfs).sort_values(['bin1_id', 'bin2_id'])

    return dict(bin1_id=m['bin1_id'].to_numpy().astype(np.int32), bin2_id=m['bin2_id'].to_numpy().astype(np.int32),
                count=m['balanced'].to_numpy(), non_nan_mask=non_nan_mask)


def preprocess_sprase(sparse_data, dups='keep'):
    bin1_id, bin2_id, count = sparse_data['bin1_id'], sparse_data['bin2_id'], sparse_data['count']
    if dups != 'keep':
        _, unique_idx, inverse_idx, unique_counts = np.unique(np.vstack([bin1_id, bin2_id]), axis=1, return_index=True, return_inverse=True, return_counts=True)
        if dups == 'fix':
            bin1_id = bin1_id[unique_idx]
            bin2_id = bin2_id[unique_idx]
            counts_uniq = np.zeros(unique_idx.size)
            for c, t in zip(count, inverse_idx):
                counts_uniq[t] += c
            count = counts_uniq

        elif dups == 'ignore':
            bin1_id = bin1_id[unique_idx]
            bin2_id = bin2_id[unique_idx]
            count = counts_uniq
        elif dups == 'remove':
            idx_mask = unique_counts == 1
            bin1_id = bin1_id[unique_idx[idx_mask]]
            bin2_id = bin2_id[unique_idx[idx_mask]]
            count = count[unique_idx[idx_mask]]
        else:
            raise ValueError(f"Invalid dup: {dups}")

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

def chr_select(fit, indices):
    """
    Return a copy of the given fit, with lambdas reduced to the bins from the requested chromosomes. cis_lengths is updated accordingly.
    """
    chr_offsets = np.cumsum((0,) + tuple(fit['cis_lengths']))
    cis_lengths = [ fit['cis_lengths'][i] for i in indices ]
    slices = [ slice(chr_offsets[i], chr_offsets[i+1]) for i in indices ]
    lambdas = np.concatenate([ fit['state_probabilities'][s] for s in slices ])

    return {**fit, 'cis_lengths': cis_lengths, 'state_probabilities': lambdas }

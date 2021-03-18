import argparse
import array_utils
import hic_analysis as hic
import numpy as np
import time

def sample_without_replacement(vec, sample_size, max_chunk=1000):
    """
    Sample without replacement from vec, where vec[i] is the how many times the i-th element is present.
    sample_size can be an int, to represent an absolute number of samples, or a float that represents the
    number of samples as a fraction of vec's sum.
    """
    _vec = np.array(vec)
    sampled = np.zeros(_vec.shape)
    if isinstance(sample_size, float) and (0 <= sample_size <= 1):
        _sample_size = int(np.sum(_vec) * sample_size)
    elif isinstance(sample_size, int) and sample_size >= 0:
        _sample_size = sample_size
    else:
        raise TypeError("sample_size must be a float in [0, 1] or non-negative int")
    while _sample_size > 0:
        counts = np.cumsum(vec)
        chunk_size = np.minimum(_sample_size, max_chunk)
        choices_vec = np.random.choice(counts[-1], size=chunk_size, replace=False)
        indices = np.searchsorted(counts, choices_vec, side='right')
        indices_grouped, choices_grouped = np.unique(indices, return_counts=True)
        sampled[indices_grouped] += choices_grouped
        _vec[indices_grouped] -= choices_grouped
        _sample_size -= chunk_size
    return sampled

def downsample_matrix(hic_mat, samples):
    print("Sampling")
    downsampled_lower_tri = sample_without_replacement(array_utils.get_lower_triangle(hic_mat), samples, max_chunk=10000000)
    print("To symm")
    downsampled_mat = array_utils.triangle_to_symmetric(hic_mat.shape[0], downsampled_lower_tri, k=-1, fast=True)
    print("Balancing")
    balanced_mat = array_utils.balance(downsampled_mat, ignorezeros=True)
    
    return balanced_mat

def main():
    parser = argparse.ArgumentParser(description = 'Experiment number (diferent initial random vectors)')
    parser.add_argument('-m', help='file name', dest='filename', type=str, required=True)
    parser.add_argument('-c', help='chromosome', dest='chromosome', type=str, required=True)
    parser.add_argument('-r', help='resolution', dest='resolution', type=str, required=True)
    parser.add_argument('-o', help='output file name', dest='output', type=str, required=True)
    parser.add_argument('-s', help='samples', dest='samples', type=float, required=True)
    parser.add_argument('--seed', help='set random seed', dest='seed', type=int, required=False, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        print(f"Using seed {args.seed}")
        np.random.seed(args.seed)

    start = time.time()
    print(f"Reading matrix for chr {args.chromosome} at resolution {args.resolution} from {args.filename}")
    unbalanced = hic.get_matrix_from_coolfile(args.filename, args.resolution, args.chromosome, balance=False)
    print(f"Downsampling to {args.samples} of samples")
    downsampled = downsample_matrix(unbalanced, args.samples)
    print(f"Resetting NaN columns")
    balanced = hic.get_matrix_from_coolfile(args.filename, args.resolution, args.chromosome, balance=True)
    nans = np.isnan(balanced).all(axis=1)
    downsampled[nans, :] = downsampled[:, nans] = np.nan
    print(f'Saving into {args.output}')
    np.save(args.output, downsampled)
    end = time.time()
    elapsed = end - start
    print(f'Done. Took: {elapsed}s')

main()

import argparse
import array_utils
import hic_analysis as hic
import gc_model as gc
import numpy as np
import time

def resample_matrix(hic_mat, reads):
    interaction_probabilities = np.nan_to_num(array_utils.get_lower_triangle(array_utils.normalize_tri_l1(hic_mat)))
    resampled_reads = np.random.multinomial(reads, interaction_probabilities, size=1)
    resampled_mat = array_utils.triangle_to_symmetric(hic_mat.shape[0], resampled_reads, k=-1, fast=True)

    return array_utils.balance(resampled_mat, ignorezeros=True)

def int_or_float(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def main():
    parser = argparse.ArgumentParser(description = 'Experiment number (diferent initial random vectors)')
    parser.add_argument('--reads', help='reads (number or ratio from -m)', dest='reads', type=int_or_float, required=False, default=100.0)
    parser.add_argument('-m', help='file name', dest='filename', type=str, required=False)
    parser.add_argument('-c', help='chromosome', dest='chromosome', type=str, required=False)
    parser.add_argument('-r', help='resolution', dest='resolution', type=str, required=False)
    parser.add_argument('-f', help='fit filename', dest='fit', type=str, required=True)
    parser.add_argument('-o', help='output file name', dest='output', type=str, required=True)
    parser.add_argument('--seed', help='set random seed', dest='seed', type=int, required=False, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        print(f"Using seed {args.seed}")
        np.random.seed(args.seed)

    start = time.time()
    if isinstance(args.reads, int) :
        if args.reads < 0:
            raise RuntimeError("--reads must be positive")
        print(f"Using {args.reads} reads from command line")
        reads = args.reads
    else: # args.reads is float
        if args.filename is None or args.chromosome is None or args.resolution is None:
            raise RuntimeError("either -m, -c and -r must be passed or --reads must be passed an integer")
        if args.reads  < 0 or args.reads > 1:
            raise RuntimeError("--reads ratio must be between 0 and 1")
        print(f"Using {args.reads:.2%} of reads of chr {args.chromosome} at resolution {args.resolution} from {args.filename}")
        unbalanced = hic.get_matrix_from_coolfile(args.filename, args.resolution, args.chromosome, balance=False)
        total_reads = np.nansum(array_utils.get_lower_triangle(unbalanced))
        reads = total_reads * args.reads
        print(f"Data has {total_reads} reads, using {reads}")

    print(f"Reading fit from {args.fit} and converting to probabilities")
    fit = np.load(args.fit, allow_pickle=True)['parameters'][()]
    fit_mat = gc.generate_interactions_matrix(**fit)
    print(f"Resampling from fit using {reads} reads")
    resampled = resample_matrix(fit_mat, reads)
    print(f"Resetting NaN columns")
    nans = np.isnan(fit['state_probabilities']).all(axis=1)
    resampled[nans, :] = resampled[:, nans] = np.nan
    print(f'Saving into {args.output}')
    np.save(args.output, resampled)
    end = time.time()
    elapsed = end - start
    print(f'Done. Took: {elapsed}s')

if __name__ == '__main__':
    main()

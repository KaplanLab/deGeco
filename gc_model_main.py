import argparse
import os
import sys
import time
import numpy as np

from gc_model import fit
from hic_analysis import get_matrix_from_coolfile
from array_utils import balance

def detect_file_type(filename):
    if filename.endswith('.mcool'):
        return 'mcool'
    elif filename.endswith('.npy'):
        return 'numpy'
    else:
        raise ValueError('unknown file type')

def main():
    #################################################################################################
    # Calculating the maximal likelihood probabilities vector and distance decay power of the model #
    #################################################################################################

    parser = argparse.ArgumentParser(description = 'Experiment number (diferent initial random vectors)')
    parser.add_argument('-m', help='file name', dest='filename', type=str, required=True)
    parser.add_argument('-t', help='file type', dest='type', type=str, choices=['mcool, numpy, auto'], default='auto')
    parser.add_argument('-o', help='output file name', dest='output', type=str, required=False, default='gc_out.npz')
    parser.add_argument('-ch', help='chromosome (required for type=mcool)', dest='chrom', type=str, required=False)
    parser.add_argument('-kb', help='resolution (required for type=mcool)', dest='resolution', type=int, required=False)
    parser.add_argument('-n', help='number of states', dest='nstates', type=int, required=False, default=2)
    parser.add_argument('-s', help='shape of weights matrix', dest='shape', type=str, required=False, default='symmetric')
    parser.add_argument('-b', help='balance matrix before fitting', dest='balance', type=bool, required=False, default=False)
    parser.add_argument('--seed', help='set random seed', dest='seed', type=int, required=False, default=None)
    args = parser.parse_args()
    
    filename = args.filename
    file_type = args.type
    if file_type == 'auto':
        file_type = detect_file_type(filename)
    if file_type == 'mcool' and (args.chrom is None or args.resolution is None):
        print("chromosome and resolution must be given for mcool files")
        sys.exit(1)
    output_file = args.output
    nstates = args.nstates
    shape = args.shape
    if file_type == 'mcool':
        chrom = f'chr{args.chrom}'
        experiment_resolution = args.resolution
        interactions_mat = lambda: get_matrix_from_coolfile(filename, experiment_resolution, chrom)
    else:
        interactions_mat = lambda: np.load(filename)

    if args.balance:
        interactions_mat = lambda: balance(interactions_mat())

    if args.seed is not None:
        print(f'Setting random seed to {args.seed}')
        np.random.seed(args.seed)

    print(f'Fitting {filename} to model with {nstates} states and weight shape {shape}. Balance = {args.balance}.')
    start_time = time.time()
    probabilities_vector, state_weights, distance_decay_power_value = fit(interactions_mat(), number_of_states=nstates,
            weights_shape=shape)
    end_time = time.time()
    print(f'Took {end_time - start_time} seconds')

    np.savez_compressed(output_file, lambdas=probabilities_vector, weights=state_weights, alpha=distance_decay_power_value, 
            seed=args.seed)
    print(f'Data saved into {output_file} in npz format.')
 
if __name__ == '__main__':
    main()

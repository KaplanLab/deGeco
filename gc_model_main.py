import argparse
import os
import sys
import time
import numpy as np
import itertools

from gc_model import fit
import gc_model
from hic_analysis import get_matrix_from_coolfile, get_chr_lengths
from array_utils import balance

def detect_file_type(filename):
    if filename.endswith('.mcool'):
        return 'mcool'
    elif filename.endswith('.npy'):
        return 'numpy'
    else:
        raise ValueError('unknown file type')

def parse_keyvalue(s):
    d = {}
    if s:
        args_list = s.split(',')
        for a in args_list:
            k, _, raw_v = a.partition('=')
            try:
                v = int(raw_v)
            except ValueError:
                try:
                    v = float(raw_v)
                except ValueError:
                    v = raw_v
            d[k] = v
    return d

def main():
    #################################################################################################
    # Calculating the maximal likelihood probabilities vector and distance decay power of the model #
    #################################################################################################

    parser = argparse.ArgumentParser(description = 'Experiment number (diferent initial random vectors)')
    parser.add_argument('-m', help='file name', dest='filename', type=str, required=True)
    parser.add_argument('-t', help='file type', dest='type', type=str, choices=['mcool, numpy, auto'], default='auto')
    parser.add_argument('-o', help='output file name', dest='output', type=str, required=False, default='gc_out.npz')
    parser.add_argument('-ch', help='chromosome (required for type=mcool). format: chrX[,chrY]', dest='chrom',
            type=str, required=False)
    parser.add_argument('-kb', help='resolution (required for type=mcool)', dest='resolution', type=int, required=False)
    parser.add_argument('-n', help='number of states', dest='nstates', type=int, required=False, default=2)
    parser.add_argument('-s', help='shape of weights matrix', dest='shape', type=str, required=False, default='diag')
    parser.add_argument('-b', help='balance matrix before fitting', dest='balance', type=bool, required=False, default=False)
    parser.add_argument('--seed', help='set random seed. If comma separated, will be used per iteration', dest='seed', type=int, nargs='+', required=False, default=None)
    parser.add_argument('--init', help='solution to init by', dest='init', type=str, required=False, default=None)
    parser.add_argument('--iterations', help='number of times to run the model before choosing the best solution', dest='iterations', type=int, required=False, default=10)
    parser.add_argument('--optimize-args', help='Override optimization args, comma-separated key=value', dest='optimize', type=str, required=False, default='')
    parser.add_argument('--kwargs', help='additional args, comma-separated key=value', dest='kwargs', type=str, required=False, default='')
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
        if args.chrom == 'all':
            chroms = [ 'all' ]
        else:
            format_chr = lambda c : f'chr{c}' if not str(c).startswith('chr') else str(c)
            chroms = [ format_chr(x) for x in args.chrom.split(',') ]
        experiment_resolution = args.resolution
        cis_lengths = get_chr_lengths(filename, experiment_resolution, chroms)
        interactions_mat = lambda: get_matrix_from_coolfile(filename, experiment_resolution, *chroms)
    else:
        cis_lengths = None
        interactions_mat = lambda: np.load(filename)

    if args.balance:
        print('Will balance the matrix before fit')
        interactions_mat_unbalanced = interactions_mat
        interactions_mat = lambda: balance(interactions_mat_unbalanced())

    optimize_options = parse_keyvalue(args.optimize)
    print(f"Optimize overrides: {optimize_options}")

    kwargs = parse_keyvalue(args.kwargs)
    print(f"Adding kwargs: {kwargs}")

    init_values = {}
    if args.init:
        print(f'Using {args.init} to init fit')
        init_values = np.load(args.init)['parameters']

    print(f'Fitting {filename} to model with {nstates} states and weight shape {shape}')
    durations = []
    best_score = np.inf
    best_args = None
    start_time = time.time()
    for i, s in itertools.zip_longest(range(args.iterations), args.seed):
        print(f"* Starting iteration number {i+1}")
        if s is not None:
            print(f'** Setting random seed to {s}')
            np.random.seed(s)
        ret = fit(interactions_mat(), number_of_states=nstates, weights_shape=shape, init_values=init_values, cis_lengths=cis_lengths,
                    optimize_options=optimize_options, **kwargs)
        end_time = time.time()
        ret_score = ret[-1].fun
        if ret_score < best_score:
            print(f"** Changing best solution to iteration {i+1}")
            best_score = ret_score
            best_args = ret
        durations.append(end_time - start_time)
        start_time = end_time
    assert best_args is not None
    probabilities_vector, state_weights, cis_dd_power, trans_dd, optimize_result = best_args
    print(f'Time per iteration was: {durations} seconds')

    model_params = dict(state_probabilities=probabilities_vector, state_weights=state_weights, cis_dd_power=cis_dd_power, trans_dd=trans_dd, cis_lengths=cis_lengths)
    metadata = dict(optimize_success=optimize_result.success, optimize_iterations=optimize_result.nit, optimize_value=optimize_result.fun, args=vars(args),
            durations=durations)
    np.savez_compressed(output_file, parameters=model_params, metadata=metadata)
    print(f'Data saved into {output_file} in npz format.')
 
if __name__ == '__main__':
    main()

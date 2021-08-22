import runpy
import argparse
import os
import sys
import time
import numpy as np
import itertools

from gc_model import fit
import gc_model
import gc_datafile
from hic_analysis import get_matrix_from_coolfile, get_sparse_matrix_from_coolfile, get_chr_lengths, preprocess_sprase
from array_utils import balance
from zero_sampler import ZeroSampler

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

def read_functions(filename):
    additional_kwargs = dict()
    if filename is None:
        return additional_kwargs
    functions_module = runpy.run_path(filename)
    
    lambdas_hyper = functions_module.get('lambdas_hyper')
    if lambdas_hyper is not None:
        additional_kwargs['lambdas_hyper'] = lambdas_hyper

    regularization = functions_module.get('regularization')
    if regularization is not None:
        additional_kwargs['regularization'] = regularization

    return additional_kwargs

def main():
    parser = argparse.ArgumentParser(description = 'Experiment number (diferent initial random vectors)')
    parser.add_argument('-m', help='file name', dest='filename', type=str, required=True)
    parser.add_argument('-t', help='file type', dest='type', type=str, choices=['mcool, numpy, auto'], default='auto')
    parser.add_argument('-o', help='output file name', dest='output', type=str, required=False, default='gc_out.npz')
    parser.add_argument('-ch', help='chromosome (required for type=mcool). format: chrX[,chrY]', dest='chrom',
            type=str, required=False)
    parser.add_argument('-kb', help='resolution (required for type=mcool)', dest='resolution', type=int, required=False)
    parser.add_argument('-n', help='number of states', dest='nstates', type=int, required=False, default=2)
    parser.add_argument('-s', help='shape of weights matrix. Format shape[,trans_shape]', dest='shape', type=str, required=False, default='diag,diag')
    parser.add_argument('-b', help='balance matrix before fitting', dest='balance', type=bool, required=False, default=False)
    parser.add_argument('--seed', help='set random seed. If comma separated, will be used per iteration', dest='seed', type=int, nargs='+', required=False, default=None)
    parser.add_argument('--init', help='solution to init by', dest='init', type=str, required=False, default=None)
    parser.add_argument('--iterations', help='number of times to run the model before choosing the best solution', dest='iterations', type=int, required=False, default=10)
    parser.add_argument('--functions', help='Python file that includes regularization or lambdas_hyper functions', dest='functions', type=str, required=False, default=None)
    parser.add_argument('--sparse', help='Use sparse model', dest='sparse', action='store_true', default=False)
    parser.add_argument('--zero-sample', help='Number of zeros to sample', dest='zero_sample', type=int, default=None)
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
    if args.sparse and args.zero_sample is None:
        print("--zero-sample must be specified when using sparse model")
        sys.exit(1)
    output_file = args.output
    nstates = args.nstates
    if ',' in args.shape:
        cis_shape, trans_shape = args.shape.split(',')
    else:
        cis_shape = trans_shape = args.shape
    if file_type == 'mcool':
        if args.chrom == 'all':
            chroms = [ 'all' ]
        else:
            format_chr = lambda c : f'chr{c}' if not str(c).startswith('chr') else str(c)
            chroms = [ format_chr(x) for x in args.chrom.split(',') ]
        experiment_resolution = args.resolution
        cis_lengths = get_chr_lengths(filename, experiment_resolution, chroms)
        if args.sparse:
            interactions_mat = preprocess_sprase(get_sparse_matrix_from_coolfile(filename, experiment_resolution, *chroms))
        else:
            interactions_mat = lambda: get_matrix_from_coolfile(filename, experiment_resolution, *chroms)
    else:
        cis_lengths = None
        interactions_mat = lambda: np.load(filename)
        experiment_resolution = args.resolution or 1

    if args.balance:
        print('Will balance the matrix before fit')
        interactions_mat_unbalanced = interactions_mat
        interactions_mat = lambda: balance(interactions_mat_unbalanced())

    functions_options = read_functions(args.functions)
    print(f"Passing functions {list(functions_options.keys())}")

    optimize_options = parse_keyvalue(args.optimize)
    print(f"Optimize overrides: {optimize_options}")

    kwargs = parse_keyvalue(args.kwargs)
    print(f"Adding kwargs: {kwargs}")

    init_values = {}
    if args.init:
        print(f'Using {args.init} to init fit')
        init_values = gc_datafile.load(args.init)['parameters']

    fit_args = dict(number_of_states=nstates, cis_weights_shape=cis_shape, trans_weights_shape=trans_shape, init_values=init_values, cis_lengths=cis_lengths,
            optimize_options=optimize_options, resolution=1)
    print(f'Fitting {filename} to model with {nstates} states and weight shape {cis_shape},{trans_shape}')
    durations = []
    best_score = np.inf
    best_args = None
    start_time = time.time()
    if args.sparse:
        nbins = interactions_mat['non_nan_mask'].shape[0]
        nn_mask_int = interactions_mat['non_nan_mask'].astype('int8') # Workaround Cython not working with bool arrays
        zero_sampler = ZeroSampler(nbins, interactions_mat['bin1_id'], interactions_mat['bin2_id'], nn_mask_int)
    for i, s in itertools.zip_longest(range(args.iterations), args.seed):
        print(f"* Starting iteration number {i+1}")
        if s is not None:
            print(f'** Setting random seed to {s}')
            np.random.seed(s)
        if args.sparse:
            print(f'** Sampling {args.zero_sample} zeros')
            sampled_zeros = zero_sampler.sample_zeros(args.zero_sample)
            ret = gc_model.fit_sparse(interactions_mat, z_const_idx=sampled_zeros, z_count=zero_sampler.zero_count,
                    **fit_args, **functions_options, **kwargs)
        else:
            ret = gc_model.fit(interactions_mat(), **fit_args, **functions_options, **kwargs)
        end_time = time.time()
        ret_score = ret[-1].fun
        if ret_score < best_score:
            print(f"** Changing best solution to iteration {i+1}")
            best_score = ret_score
            best_args = ret
        durations.append(end_time - start_time)
        start_time = end_time
    assert best_args is not None
    probabilities_vector, cis_weights, trans_weights, cis_dd_power, trans_dd, optimize_result = best_args
    print(f'Time per iteration was: {durations} seconds')

    model_params = dict(state_probabilities=probabilities_vector, cis_weights=cis_weights, trans_weights=trans_weights,
            cis_dd_power=cis_dd_power, trans_dd=trans_dd, cis_lengths=cis_lengths)
    metadata = dict(optimize_success=optimize_result.success, optimize_iterations=optimize_result.nit,
            optimize_value=optimize_result.fun, args=vars(args), durations=durations)
    gc_datafile.save(output_file, parameters=model_params, metadata=metadata)
    print(f'Data saved into {output_file} in npz format.')
 
if __name__ == '__main__':
    main()

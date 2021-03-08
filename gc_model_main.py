import argparse
import os
import sys
import time
import numpy as np
import itertools

from gc_model import fit
import gc_model
from hic_analysis import get_matrix_from_coolfile, get_chr_lengths
from array_utils import balance, get_lower_triangle

def detect_file_type(filename):
    if filename.endswith('.mcool'):
        return 'mcool'
    elif filename.endswith('.npy'):
        return 'numpy'
    else:
        raise ValueError('unknown file type')

def fit_iterate(matrix, init, rounds=5, **kwargs):
    init_l, init_w, init_a = init['lambdas'], init['weights'], init['alpha']
    if np.ndim(init_l) > 1:
        init_l = init_l[~np.isnan(init_l).all(axis=1)]
    if np.ndim(init_w) > 1:
        init_w = get_lower_triangle(init_w, k=0)
    params = dict(lambdas=init_l, weights=init_w, alpha=init_a)
    for i in range(rounds):
        for (k1, k2) in itertools.combinations(('lambdas', 'weights', 'alpha'), 2):
            v1, v2 = params[k1], params[k2]
            fixed_fit = gc_model.fit(matrix, fixed_values={k1: v1, k2: v2}, init_values=params, **kwargs)
            params['lambdas'], params['weights'], params['alpha'] = fixed_fit[:3]
            params['lambdas'] = params['lambdas'][~np.isnan(params['lambdas']).all(axis=1)]
            params['weights'] = get_lower_triangle(params['weights'], k=0)
            print(i, params['weights'])
    return fixed_fit

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
    parser.add_argument('--seed', help='set random seed', dest='seed', type=int, required=False, default=None)
    parser.add_argument('--init', help='solution to init by', dest='init', type=str, required=False, default=None)
    parser.add_argument('--rounds', help='rounds of fixed-fit', dest='rounds', type=int, required=False, default=0)
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
        interactions_mat_unbalanced = interactions_mat
        interactions_mat = lambda: balance(interactions_mat_unbalanced())

    if args.seed is not None:
        print(f'Setting random seed to {args.seed}')
        np.random.seed(args.seed)

    optimize_options = parse_keyvalue(args.optimize)
    print(f"Optimize overrides: {optimize_options}")

    kwargs = parse_keyvalue(args.kwargs)
    print(f"Adding kwargs: {kwargs}")

    if args.rounds == 0:
        init_fit = {}
        if args.init:
            print(f'Using {args.init} to init fit')
            init_fit = np.load(args.init)
        fit_func = lambda **kwargs: fit(interactions_mat(), init_values=init_fit, **kwargs)
    else:
        print(f"Fixing fit over {args.rounds} rounds")
        non_nan_mask = ~np.isnan(interactions_mat()).all(1)
        _, lambdas_param_count = gc_model.lambdas_hyper_default(non_nan_mask, nstates)
        _, weights_param_count = gc_model.weight_hyperparams(shape, nstates)
        v = gc_model.init_variables(lambdas_param_count, weights_param_count)
        fix_init = dict(lambdas=v[:-weights_param_count], weights=v[-weights_param_count:], alpha=-1)
        if args.init:
            print(f'Using {args.init} to init fit')
            init_file = np.load(args.init)
            for k in ('lambdas', 'weights', 'alpha'):
                if k in init_file:
                    fix_init[k] = init_file[k]
        fit_func = lambda **kwargs: fit_iterate(interactions_mat(), fix_init, args.rounds, **kwargs)

    print(f'Fitting {filename} to model with {nstates} states and weight shape {shape}. Balance = {args.balance}.')
    start_time = time.time()
    probabilities_vector, state_weights, cis_dd_power, trans_dd, optimize_result = \
            fit_func(number_of_states=nstates, weights_shape=shape, cis_lengths=cis_lengths,
                    optimize_options=optimize_options, **kwargs)
    end_time = time.time()
    print(f'Took {end_time - start_time} seconds')

    model_params = dict(state_probabilities=probabilities_vector, state_weights=state_weights, cis_dd_power=cis_dd_power, trans_dd=trans_dd, cis_lengths=cis_lengths)
    metadata = dict(optimize_success=optimize_result.success, optimize_iterations=optimize_result.nit, optimize_value=optimize_result.fun, args=vars(args))
    np.savez_compressed(output_file, parameters=model_params, metadata=metadata)
    print(f'Data saved into {output_file} in npz format.')
 
if __name__ == '__main__':
    main()

import argparse
import os
import sys
import time
import numpy as np

from gc_model import fit
from hic_analysis import get_matrix_from_coolfile

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
    parser.add_argument('-d', help='output directory name', dest='directory', type=str, required=False, default='.')
    parser.add_argument('-ch', help='chromosome (required for type=mcool)', dest='chrom', type=str, required=False)
    parser.add_argument('-kb', help='resolution (required for type=mcool)', dest='resolution', type=int, required=False)
    parser.add_argument('-n', help='number of states', dest='nstates', type=int, required=False, default=2)
    parser.add_argument('-s', help='shape of weights matrix', dest='shape', type=str, required=False, default='symmetric')
    args = parser.parse_args()
    
    filename = args.filename
    file_type = args.type
    if file_type == 'auto':
        file_type = detect_file_type(filename)
    if file_type == 'mcool' and (args.chrom is None or args.resolution is None):
        print("chromosome and resolution must be given for mcool files")
        sys.exit(1)
    output_dir = args.directory
    nstates = args.nstates
    shape = args.shape
    if file_type == 'mcool':
        chrom = f'chr{args.chrom}'
        experiment_resolution = args.resolution
        output_filename_suffix = f'_{chrom}_{experiment_resolution}kb_{nstates}states'
        interactions_mat = get_matrix_from_coolfile(filename, experiment_resolution, chrom)
    else:
        interactions_mat = np.load(filename)
        output_filename_suffix = f'_{nstates}states'

    print(f'Fitting {filename} to model with {nstates} states and weight shape {shape}.')
    non_nan_mask = ~np.isnan(interactions_mat).all(1)
    np.save(os.path.join(output_dir, f'experimented_cis_interactions_is_not_nan_vec{output_filename_suffix}.npy'), non_nan_mask)
    np.savez_compressed(os.path.join(output_dir, f'input_matrix{output_filename_suffix}'), a=interactions_mat)

    start_time = time.time()
    probabilities_vector, state_weights, distance_decay_power_value = fit(interactions_mat, number_of_states=nstates,
            weights_shape=shape)
    end_time = time.time()
    print(f'Took {end_time - start_time} seconds')

    np.save(os.path.join(output_dir, f'lambda_probabilities{output_filename_suffix}.npy'), probabilities_vector)
    np.save(os.path.join(output_dir, f'model_distance_decay_power_value{output_filename_suffix}.npy'),
            distance_decay_power_value)
    np.save(os.path.join(output_dir, f'model_state_weights{output_filename_suffix}.npy'), state_weights)

    print(f'Data saved into directory {output_dir} with suffix {output_filename_suffix}.')
 
if __name__ == '__main__':
    main()

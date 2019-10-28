from __future__ import division
import numpy as np
import scipy as sp
from scipy import optimize
import time
import argparse
import cooler
import matplotlib
import os
matplotlib.use('Agg')

start_time = time.time()  # Initial time point to measure function performances

###################################################################################################
# The "evaluation_of_likelihood_gradient" function calculates the likelihood of a given vector of #
# compartments belonging probabilities and the distance decay power value.                        #
# It also calculates the first order derivative for the gradient decent step of the algorithm.    #
###################################################################################################

def evaluation_of_likelihood_gradient(variables, *args):
    
    experimented_cis_interactions_is_not_nan_vec = args[1]
    
    arange_of_not_nan_locations = np.arange(np.size(experimented_cis_interactions_is_not_nan_vec))[experimented_cis_interactions_is_not_nan_vec]

    number_of_bins = np.size(variables) - 1  

    probabilities_vector = variables[0:number_of_bins]  

    distance_decay_power = variables[number_of_bins]  

    lower_triangular_experimented_cis_interactions = args[0] 

    lower_triangular_distance_matrix = (arange_of_not_nan_locations[None].T-arange_of_not_nan_locations)[np.tril_indices(number_of_bins,-1)]

    lower_triangular_cis_interactions_model_with_no_dd = (np.outer(probabilities_vector, probabilities_vector) + np.outer(1 - probabilities_vector, 1 - probabilities_vector))[np.tril_indices(number_of_bins,-1)]

    lower_triangular_cis_interactions_model = np.multiply(lower_triangular_distance_matrix ** distance_decay_power, lower_triangular_cis_interactions_model_with_no_dd)
    
    probability_normalizing_constant = 1.0 / np.nansum(lower_triangular_cis_interactions_model)

    gradient_of_current_distance_decay_power_likelihood = np.nansum(np.multiply(np.log(lower_triangular_distance_matrix),lower_triangular_cis_interactions_model)) * probability_normalizing_constant - np.nansum(np.multiply(lower_triangular_experimented_cis_interactions, np.log(lower_triangular_distance_matrix)))

    cis_interactions_likelihood = np.nansum(np.multiply(lower_triangular_experimented_cis_interactions, np.log(probability_normalizing_constant * lower_triangular_cis_interactions_model)))
        
    cis_interactions_model_with_no_dd = np.outer(probabilities_vector, probabilities_vector) + np.outer(1 - probabilities_vector, 1 - probabilities_vector)

    likelihood_of_current_probabilities_vector = -1.0 * cis_interactions_likelihood

    #should try to improve this line that replaces the following 5 lines:
    #gradient_of_current_probabilities_vector_likelihood = np.nansum((2.0 * lower_triangular_distance_matrix ** distance_decay_power) * (2.0 * probabilities_vector - np.ones(number_of_bins)) * probability_normalizing_constant, axis=1) - np.nansum(((lower_triangular_experimented_cis_interactions * (2.0 * probabilities_vector - np.ones(number_of_bins))) * (1.0 / cis_interactions_model_with_no_dd)), axis=1)
        
    full_distance_matrix = (np.absolute(arange_of_not_nan_locations-arange_of_not_nan_locations[None].T) + np.diag(np.zeros(number_of_bins)+np.nan))
    gradient_of_current_probabilities_vector_likelihood_part1 = np.nansum((full_distance_matrix ** distance_decay_power) * (2.0 * probabilities_vector - np.ones(number_of_bins)) * probability_normalizing_constant, axis=1) 

    full_experimented_cis_interactions = cis_interactions_model_with_no_dd*0
    full_experimented_cis_interactions[np.tril_indices(number_of_bins,-1)] = lower_triangular_experimented_cis_interactions
    full_experimented_cis_interactions = full_experimented_cis_interactions + full_experimented_cis_interactions.transpose()
    gradient_of_current_probabilities_vector_likelihood_part2 = - np.nansum(((full_experimented_cis_interactions * (2.0 * probabilities_vector - np.ones(number_of_bins))) * (1.0 / cis_interactions_model_with_no_dd)), axis=1)

    gradient_of_current_probabilities_vector_likelihood = gradient_of_current_probabilities_vector_likelihood_part1 + gradient_of_current_probabilities_vector_likelihood_part2

    final_likelihood_gradient_for_evaluation_of_likelihood_gradient_function = np.append(gradient_of_current_probabilities_vector_likelihood, [gradient_of_current_distance_decay_power_likelihood])

    final_likelihood_gradient_for_evaluation_of_likelihood_gradient_function = np.nan_to_num(final_likelihood_gradient_for_evaluation_of_likelihood_gradient_function)

    return likelihood_of_current_probabilities_vector, final_likelihood_gradient_for_evaluation_of_likelihood_gradient_function


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

def save_input_data(interactions_mat, output_dir, output_filename_suffix=''):
    """
    Save to disk the interactions matrix and the original indices of non-NaN values.
    This can be used for later correlation evaluation with bio tracks.

    The input matrix filename is 'input_matrix{suffix}.npz'
    The non-NaN indices filename is 'experimented_cis_interactions_is_not_nan_vec{suffix}.npy'

    :param numpy-array interactions_mat: Interactions (Hi-C) matrix
    :param str chromosome: chromosome used for this experiment (chrXX)
    :param int resolution: matrix resolution (bin size)
    :param str output_dir: directory where output files will be saved
    """
    non_nan_indices = ~np.isnan(interactions_mat).all(1)
    non_nan_output_filename = f'experimented_cis_interactions_is_not_nan_vec{output_filename_suffix}.npy'
    np.save(os.path.join(output_dir, non_nan_output_filename), non_nan_indices)

    input_matrix_filename = f'input_matrix{output_filename_suffix}'
    np.savez_compressed(os.path.join(output_dir, input_matrix_filename), a=interactions_mat)

def save_output_data(probabilities_vector, distance_decay_power, output_dir, output_filename_suffix):
    filename_dd_power_value = f'model_distance_decay_power_value{output_filename_suffix}.npy'
    filename_probabilities = f'lambda_probabilities{output_filename_suffix}.npy'
    np.save(os.path.join(output_data_dir, filename_dd_power_value), distance_decay_power)  
    np.save(os.path.join(output_data_dir, filename_probabilities), probabilities_vector)

def matrix_to_value_vector(input_interactions_mat):
    """
    Convert the input interactions matrix into a 1D vector of the observed values.
    The vector is taken from the lower triangle of the interactions matrix after various arrangements, cleaning and normalizations.

    :param array input_interactions_mat: Interactions matrix
    :return: A tuple of the vector of interaction values and the new number of bins (after empty ones were removed)
    :rtype: tuple
    """
    # fill the diagonal with nan values
    clean_mat = input_interactions_mat.copy()
    np.fill_diagonal(clean_mat, np.nan) 

    # delete empty rows and columns
    non_nan_indices = ~np.isnan(input_interactions_mat).all(1)
    clean_mat = clean_mat[:, non_nan_indices][non_nan_indices, :]

    # make sure we have a symmetric cis interactions matrix
    clean_mat = np.maximum(clean_mat, clean_mat.transpose())

    number_of_bins = np.size(clean_mat, 0)

    lower_triangle_indices = np.tril_indices(number_of_bins, -1)
    # normalizing the cis interaction matrix to sum of 1 for easier gradient evaluation
    clean_mat /= np.nansum(clean_mat[lower_triangle_indices]) 

    # keep only the lower triangular interaction matrix
    value_vec = clean_mat[lower_triangle_indices]

    return value_vec, number_of_bins

def process_matrix(input_interactions_mat):
    non_nan_indices = ~np.isnan(input_interactions_mat).all(1)

    original_number_of_bins = np.size(input_interactions_mat, 0) 
    values_vec, new_number_of_bins = matrix_to_value_vector(input_interactions_mat)

    # set initial random vector for the algorithm
    x0_random = np.append((np.random.rand(new_number_of_bins) / 2) + 0.25, [-1])  

    # set bound for the probability vector and distance decay power
    bnd = np.append([(0.01, 0.99)] * new_number_of_bins, [(-2, -0.5)], axis=0)  

    # calculate the maximal likelihood probabilities vector and distance decay power using the "L-BFGS-B" method
    optimize_options = dict(disp=True, ftol=1.0e-10, gtol=1e-010, eps=1e-10, maxfun=100000, maxiter=100000, maxls=100)
    res = sp.optimize.minimize(fun=evaluation_of_likelihood_gradient, x0=x0_random, args=(values_vec, non_nan_indices),
            method='L-BFGS-B', jac=True, bounds=bnd, options=optimize_options)

    # resulted probabilities vector and distance decay power of the model
    model_probabilities_vector = np.zeros(original_number_of_bins) + np.nan
    model_probabilities_vector[non_nan_indices] = res.x[0:new_number_of_bins] 
    
    model_distance_decay_power_value = res.x[new_number_of_bins]

    return model_probabilities_vector, model_distance_decay_power_value

def main():

    #################################################################################################
    # Calculating the maximal likelihood probabilities vector and distance decay power of the model #
    #################################################################################################

    parser = argparse.ArgumentParser(description = 'Experiment number (diferent initial random vectors)')
    parser.add_argument('-m',help='mcool file name',dest='mcool',type=str,required=True)
    parser.add_argument('-dr',help='directory name',dest='directory',type=str,required=True)
    parser.add_argument('-ch',help='chromosome',dest='chrom',type=str,required=True)
    parser.add_argument('-l',help='local directory path',dest='local',type=str,required=True)
    parser.add_argument('-kb',dest='resolution',type=int,required=True)
    args = parser.parse_args()
    
    local = str(args.local)
    chrom = f'chr{args.chrom}'
    directory = args.directory
    mcool_filename = args.mcool
    experiment_resolution = args.resolution
    output_data_dir = os.path.join(local, directory, 'OutputData')
    output_filename_suffix = f'_{chrom}_{experiment_resolution}kb'

    input_interactions_mat = get_matrix_from_coolfile(mcool_filename, experiment_resolution, chrom)
    save_input_data(input_interactions_mat, output_data_dir, output_filename_suffix)
    model_probabilities_vector, model_distance_decay_power_value = process_matrix(input_interactions_mat)
    save_output_data(model_probabilities_vector, model_distance_decay_power_value, output_data_dir, output_filename_suffix)

    print(f'Data saved into directory {output_data_dir} with suffix {output_filename_suffix}.')
 
if __name__ == '__main__':
    main()

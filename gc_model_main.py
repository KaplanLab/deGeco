import numpy as np
import scipy as sp
from scipy import optimize
import time
import argparse
import cooler
import matplotlib
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
    chrom = 'chr' + args.chrom 
    directory = args.directory
    mcool_filename = args.mcool
    experiment_resolution = args.resolution

    coolfile = str(mcool_filename) + '::/resolutions/' + str(experiment_resolution)
    c = cooler.Cooler(coolfile)

#    chroms = c.chromnames[:-1]


    ex = c.extent(chrom) # returns a tuple with first and last bins of specified region
    d = c.matrix()[ex[0]:ex[1],ex[0]:ex[1]] # returns a numpy matrix (balanced Hi-C)
    
    experimented_cis_interactions = d
    #xperimented_cis_interactions[experimented_cis_interactions == 0] = np.nan

    # keep the indices of nan values for later correlation evaluations with bio tracks
    experimented_cis_interactions_is_not_nan_vec = ~np.isnan(experimented_cis_interactions).all(1)
    np.save(local + str(directory) + '/OutputData/experimented_cis_interactions_is_not_nan_vec_' + str(chrom) + '_'+str(experiment_resolution) + 'kb' + '.npy', experimented_cis_interactions_is_not_nan_vec)

    np.savez_compressed(local + str(directory) + '/OutputData/input_matrix_' + str(chrom) + '_' + str(experiment_resolution) + 'kb', a=experimented_cis_interactions)

    # fill the diagonal with nan values
    np.fill_diagonal(experimented_cis_interactions, np.nan) 

    original_number_of_bins = np.size(experimented_cis_interactions, 0) 

    # delete empty rows and columns
    experimented_cis_interactions = experimented_cis_interactions[:, experimented_cis_interactions_is_not_nan_vec]  
    experimented_cis_interactions = experimented_cis_interactions[experimented_cis_interactions_is_not_nan_vec, :]

    # make sure we have a symmetric cis interactions matrix
    experimented_cis_interactions = np.maximum(experimented_cis_interactions, experimented_cis_interactions.transpose())  # generate cis interaction matrix

    # replace zeros with nan values
    #experimented_cis_interactions[experimented_cis_interactions == 0] = np.nan   

    # updated chromosome length after deleting empty rows and columns
    new_number_of_bins = np.size(experimented_cis_interactions, 0)  
    
    # normalizing the cis interaction matrix to sum of 1 for easier gradient evaluation
    experimented_cis_interactions = experimented_cis_interactions / (np.nansum(experimented_cis_interactions[np.tril_indices(new_number_of_bins, -1)]) * 1.0) 

    # keep only the lower triangular interaction matrix
    experimented_cis_interactions = experimented_cis_interactions[np.tril_indices(new_number_of_bins,-1)]

    # set initial random vector for the algorithm
    x0_random = np.append((np.random.rand(new_number_of_bins) / 2) + 0.25, [-1])  

    # set bound for the probability vector and distance decay power
    bnd = np.append([(0.01, 0.99)] * new_number_of_bins, [(-2, -0.5)], axis=0)  

    # calculate the maximal likelihood probabilities vector and distance decay power using the "L-BFGS-B" method
    res = sp.optimize.minimize(fun=evaluation_of_likelihood_gradient, x0=x0_random, args=(experimented_cis_interactions,experimented_cis_interactions_is_not_nan_vec),method='L-BFGS-B',jac=True, bounds=bnd,
                               #options={'disp': True})  # calculate maximum likelihood variables vector
                               options={'disp': True,'ftol': 1.0e-10, 'gtol': 1e-010, 'eps': 1e-10, 'maxfun': 100000, 'maxiter': 100000, 'maxls': 100})  # calculate maximum likelihood variables vecto


    # resulted probabilities vector and distance decay power of the model
    model_probabilities_vector = np.zeros(original_number_of_bins) + np.nan
    model_probabilities_vector[experimented_cis_interactions_is_not_nan_vec] = res.x[0:new_number_of_bins] 
    
    model_distance_decay_power_vavlue = res.x[new_number_of_bins]

    np.save(local + str(directory) + '/OutputData/model_distance_decay_power_value_' + str(chrom) + '_' + str(experiment_resolution) + 'kb.npy', model_distance_decay_power_vavlue)  
    np.save(local + str(directory) + '/OutputData/lambda_probabilities_' + str(chrom) + '_' + str(experiment_resolution) + 'kb.npy', model_probabilities_vector)
 
main()
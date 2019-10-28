import numpy as np
from numpy import inf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from scipy import stats
import cooler

worb_cmap=matplotlib.colors.LinearSegmentedColormap.from_list('worb',colors=['white','orange','red',[0.5,0,0],'black'])
worb_cmap.set_bad([0.82,0.82,0.82])
plt.register_cmap(cmap=worb_cmap)

###################################################################################
# The "get_analysis_results" function return the analysis parameters of the model #
###################################################################################
 
def get_analysis_results(chrom, directory, experiment_resolution, local):   
    
    loaded = np.load(local + str(directory) + '/OutputData/input_matrix_' + str(chrom) + '_' + str(experiment_resolution) + 'kb.npz')
    experimented_cis_interactions = loaded['a']
	    
    # fill the diagonal with nan values
    np.fill_diagonal(experimented_cis_interactions, np.nan) 

    # replace zeros with nan values
    experimented_cis_interactions[experimented_cis_interactions == 0] = np.nan  

    # chromosome length after deleting empty rows and columns
    number_of_bins = np.size(experimented_cis_interactions, 0)  
    
    # normalizing the cis interaction matrix to sum of 1 for easier gradient evaluation
    experimented_cis_interactions = experimented_cis_interactions / (np.nansum(experimented_cis_interactions[np.tril_indices(number_of_bins, -1)]) * 1.0) 

    model_probabilities_vector = np.load(local + str(directory) + '/OutputData/lambda_probabilities_' + str(chrom) + '_' + str(experiment_resolution) + 'kb.npy')
    model_distance_decay_power_value = np.load(local + str(directory) + '/OutputData/model_distance_decay_power_value_' + str(chrom) + '_' + str(experiment_resolution) + 'kb.npy')                                        

    model_reconstruction_with_no_dd = np.outer(model_probabilities_vector,model_probabilities_vector) + np.outer(1 - model_probabilities_vector, 1 - model_probabilities_vector)

    distance_matrix = np.absolute(np.outer(np.arange(number_of_bins), np.ones(number_of_bins)) - np.arange(number_of_bins))
    np.fill_diagonal(distance_matrix, np.nan)
    
    distance_decay_matrix = distance_matrix**model_distance_decay_power_value

    model_reconstruction_with_dd = distance_decay_matrix*model_reconstruction_with_no_dd
    
    model_reconstruction_with_dd = model_reconstruction_with_dd / (np.nansum(model_reconstruction_with_dd[np.tril_indices(number_of_bins, -1)]) * 1.0)

    dd_normalized_experimented_cis_interactions = np.copy(experimented_cis_interactions) * 1.0  

    # normalizing the cis interaction matrix with similar diagonals sum
    for i in range(1, number_of_bins):
        if np.count_nonzero(~np.isnan(dd_normalized_experimented_cis_interactions[(range(number_of_bins - i), range(i, number_of_bins))])) > 0:
            dd_normalized_experimented_cis_interactions[(range(number_of_bins - i), range(i, number_of_bins))] /= 1.0 * np.nanmean(
                dd_normalized_experimented_cis_interactions[(range(number_of_bins - i), range(i, number_of_bins))])
        if np.count_nonzero(~np.isnan(dd_normalized_experimented_cis_interactions[(range(i, number_of_bins), range(number_of_bins - i))])) > 0:
            dd_normalized_experimented_cis_interactions[(range(i, number_of_bins), range(number_of_bins - i))] /= 1.0 * np.nanmean(
                dd_normalized_experimented_cis_interactions[(range(i, number_of_bins), range(number_of_bins - i))])

    dd_normalized_experimented_cis_interactions = dd_normalized_experimented_cis_interactions / (np.nansum(dd_normalized_experimented_cis_interactions[np.tril_indices(number_of_bins, -1)]) * 1.0) #normalizing to sum 1    
       
    dd_normalized_model_reconstruction = np.copy(model_reconstruction_with_dd) * 1.0  
    
    # normalizing the cis interaction matrix with similar diagonals sum
    for i in range(1, number_of_bins):
        if np.count_nonzero(~np.isnan(dd_normalized_model_reconstruction[(range(number_of_bins - i), range(i, number_of_bins))])) > 0:
            dd_normalized_model_reconstruction[(range(number_of_bins - i), range(i, number_of_bins))] /= 1.0 * np.nanmean(
                dd_normalized_model_reconstruction[(range(number_of_bins - i), range(i, number_of_bins))])
        if np.count_nonzero(~np.isnan(dd_normalized_model_reconstruction[(range(i, number_of_bins), range(number_of_bins - i))])) > 0:
            dd_normalized_model_reconstruction[(range(i, number_of_bins), range(number_of_bins - i))] /= 1.0 * np.nanmean(
                dd_normalized_model_reconstruction[(range(i, number_of_bins), range(number_of_bins - i))])
    
    dd_normalized_model_reconstruction = dd_normalized_model_reconstruction / (np.nansum(dd_normalized_model_reconstruction[np.tril_indices(number_of_bins, -1)]) * 1.0) #normalizing to sum 1    

    return experimented_cis_interactions,model_probabilities_vector,model_distance_decay_power_value, model_reconstruction_with_dd, model_reconstruction_with_no_dd, dd_normalized_experimented_cis_interactions,dd_normalized_model_reconstruction

###################################################################################
# The "main" function plots the results analysis of the genome compartments model #
###################################################################################

def main():

    ################
    ## FIGURES 1a ##
    ################
    
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


    experimented_cis_interactions,model_probabilities_vector,model_distance_decay_power_value, model_reconstruction_with_dd, model_reconstruction_with_no_dd, dd_normalized_experimented_cis_interactions,dd_normalized_model_reconstruction = get_analysis_results(chrom, directory, experiment_resolution, local)

    number_of_bins = np.size(experimented_cis_interactions,1)
    
    data_minus_model = experimented_cis_interactions-model_reconstruction_with_dd

    plt.figure()
    plt.imshow(data_minus_model, vmin=-np.nanmax(model_reconstruction_with_dd), vmax=np.nanmax(model_reconstruction_with_dd),cmap='seismic')
    plt.savefig(local + str(directory) + '/Figures/data_minus_model_' + chrom + '_' + str(experiment_resolution) + 'kb.png', dpi=1000)
    plt.close()
    np.savez_compressed(local + str(directory) + '/OutputData/data_minus_model_' + chrom + '_' + str(experiment_resolution) + 'kb', a=data_minus_model)

    normalized_data_minus_normalized_model = dd_normalized_experimented_cis_interactions-dd_normalized_model_reconstruction

    plt.figure()
    plt.imshow(normalized_data_minus_normalized_model, vmin=-np.nanmax(dd_normalized_model_reconstruction), vmax=np.nanmax(dd_normalized_model_reconstruction),cmap='seismic')
    plt.savefig(local + str(directory) + '/Figures/normalized_data_minus_normalized_model_' + chrom + '_' + str(experiment_resolution) + 'kb.png', dpi=1000)
    plt.close()
    np.savez_compressed(local + str(directory) + '/OutputData/normalized_data_minus_normalized_model_' + chrom + '_' + str(experiment_resolution) + 'kb', a=normalized_data_minus_normalized_model)

    experimented_cis_interactions[experimented_cis_interactions == 0] = np.nan
    model_reconstruction_with_dd[model_reconstruction_with_dd == 0 ] = np.nan

    log_data_over_model = np.log(experimented_cis_interactions / model_reconstruction_with_dd)

    plt.figure()
    plt.imshow(np.nan_to_num(log_data_over_model), vmin=-np.nanmax(model_reconstruction_with_dd), vmax=np.nanmax(model_reconstruction_with_dd),cmap='seismic')
    plt.savefig(local + str(directory) + '/Figures/log_data_over_model_' + chrom + '_' + str(experiment_resolution) + 'kb.png', dpi=1000)
    plt.close()
    np.savez_compressed(local + str(directory) + '/OutputData/log_data_over_model_' + chrom + '_' + str(experiment_resolution) + 'kb', a=log_data_over_model)

    dd_normalized_experimented_cis_interactions[dd_normalized_experimented_cis_interactions == 0] = np.nan
    dd_normalized_model_reconstruction[dd_normalized_model_reconstruction == 0] = np.nan

    log_normalized_data_over_normalized_model = np.log(dd_normalized_experimented_cis_interactions / dd_normalized_model_reconstruction)  

    plt.figure()
    plt.imshow(np.nan_to_num(log_normalized_data_over_normalized_model), vmin=-np.nanmax(dd_normalized_model_reconstruction), vmax=np.nanmax(dd_normalized_model_reconstruction),cmap='seismic')
    plt.savefig(local + str(directory) + '/Figures/log_normalized_data_over_normalized_model_' + chrom + '_' + str(experiment_resolution) + 'kb.png', dpi=1000)
    plt.close()
    np.savez_compressed(local + str(directory) + '/OutputData/log_normalized_data_over_normalized_model_' + chrom + '_' + str(experiment_resolution) + 'kb', a=log_normalized_data_over_normalized_model)  

    half_half_data_vs_reconstructed_model = experimented_cis_interactions
    half_half_data_vs_reconstructed_model[np.tril_indices(number_of_bins, -1)] = model_reconstruction_with_dd[np.tril_indices(number_of_bins, -1)]

    np.savez_compressed(local + str(directory) + '/OutputData/half_half_data_vs_reconstructed_model_' + chrom + '_' + str(experiment_resolution) + 'kb', a=half_half_data_vs_reconstructed_model)

    log_experimented_cis_interactions = np.log(experimented_cis_interactions)
    log_experimented_cis_interactions= log_experimented_cis_interactions - np.nanmean(log_experimented_cis_interactions)
    log_experimented_cis_interactions = log_experimented_cis_interactions / np.nanvar(log_experimented_cis_interactions)

    log_model_reconstruction_with_dd = np.log(model_reconstruction_with_dd)
    log_model_reconstruction_with_dd = log_model_reconstruction_with_dd - np.nanmean(log_model_reconstruction_with_dd)
    log_model_reconstruction_with_dd = log_model_reconstruction_with_dd / np.nanvar(log_model_reconstruction_with_dd)

    half_half_log_data_vs_reconstructed_model = log_experimented_cis_interactions
    half_half_log_data_vs_reconstructed_model[np.tril_indices(number_of_bins, -1)] = log_model_reconstruction_with_dd[np.tril_indices(number_of_bins, -1)]

    half_half_log_data_vs_reconstructed_model = half_half_log_data_vs_reconstructed_model - np.nanmin(half_half_log_data_vs_reconstructed_model)
    half_half_log_data_vs_reconstructed_model = np.nan_to_num(half_half_log_data_vs_reconstructed_model)

    plt.figure()
    plt.imshow(half_half_log_data_vs_reconstructed_model,cmap='worb')
    plt.savefig(local + str(directory) + '/Figures/half_half_log_data_vs_reconstructed_model_' + chrom + '_' + str(experiment_resolution) + 'kb.png', dpi=1000)
    plt.close()
    
    model_reconstruction_with_no_dd[model_reconstruction_with_no_dd == 0] = np.nan

    half_half_normalized_data_vs_model_compartments = dd_normalized_experimented_cis_interactions
    half_half_normalized_data_vs_model_compartments[np.tril_indices(number_of_bins, -1)] = model_reconstruction_with_no_dd[np.tril_indices(number_of_bins, -1)] / (np.nansum(model_reconstruction_with_no_dd[np.tril_indices(number_of_bins, -1)]) * 1.0)

    np.savez_compressed(local + str(directory) + '/OutputData/half_half_normalized_data_vs_model_compartments_' + chrom + '_' + str(experiment_resolution) + 'kb', a=half_half_normalized_data_vs_model_compartments)
    
    log_dd_normalized_experimented_cis_interactions = np.log(dd_normalized_experimented_cis_interactions)
    log_dd_normalized_experimented_cis_interactions = log_dd_normalized_experimented_cis_interactions  - np.nanmean(log_dd_normalized_experimented_cis_interactions )
    log_dd_normalized_experimented_cis_interactions  = log_dd_normalized_experimented_cis_interactions  / np.nanvar(log_dd_normalized_experimented_cis_interactions )

    log_model_reconstruction_with_no_dd = np.log(model_reconstruction_with_no_dd)
    log_model_reconstruction_with_no_dd = log_model_reconstruction_with_no_dd - np.nanmean(log_model_reconstruction_with_no_dd)
    log_model_reconstruction_with_no_dd = log_model_reconstruction_with_no_dd / np.nanvar(log_model_reconstruction_with_no_dd)

    half_half_log_normalized_data_vs_model_compartments = log_dd_normalized_experimented_cis_interactions
    half_half_log_normalized_data_vs_model_compartments[np.tril_indices(number_of_bins, -1)] = log_model_reconstruction_with_no_dd[np.tril_indices(number_of_bins, -1)]

    half_half_log_normalized_data_vs_model_compartments = half_half_log_normalized_data_vs_model_compartments - np.nanmin(half_half_log_normalized_data_vs_model_compartments)
    half_half_log_normalized_data_vs_model_compartments = np.nan_to_num(half_half_log_normalized_data_vs_model_compartments)

    plt.figure()
    plt.imshow(half_half_log_normalized_data_vs_model_compartments,cmap='worb')
    plt.savefig(local + str(directory) + '/Figures/half_half_log_normalized_data_vs_reconstructed_model_compartments_' + chrom + '_' + str(experiment_resolution) + 'kb.png', dpi=1000)
    plt.close()
    
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################  
    

    model_probabilities_vector_with_nan = np.load(local + str(directory) + '/OutputData/lambda_probabilities_' + str(chrom) + '_' + str(experiment_resolution) + 'kb.npy')
    model_probabilities_vector_is_not_nan_vec = np.load(local + str(directory) + '/OutputData/experimented_cis_interactions_is_not_nan_vec_' + str(chrom) + '_'+str(experiment_resolution) + 'kb.npy')
    
    model_probabilities_vector_without_nan = model_probabilities_vector_with_nan[model_probabilities_vector_is_not_nan_vec]
                                                        
    right_compartment_25_precentile  =stats.scoreatpercentile(model_probabilities_vector_without_nan[model_probabilities_vector_without_nan>0.5], 25)
    right_compartment_75_precentile  =stats.scoreatpercentile(model_probabilities_vector_without_nan[model_probabilities_vector_without_nan>0.5], 75)
    
    #right_compartment_tvar = stats.tvar(model_probabilities_vector_without_nan, (right_compartment_25_precentile,right_compartment_75_precentile))
    
    left_compartment_25_precentile  =stats.scoreatpercentile(model_probabilities_vector_without_nan[model_probabilities_vector_without_nan<0.5], 25)
    left_compartment_75_precentile  =stats.scoreatpercentile(model_probabilities_vector_without_nan[model_probabilities_vector_without_nan<0.5], 75)
    
    #left_compartment_tvar = stats.tvar(model_probabilities_vector_without_nan, (left_compartment_25_precentile,left_compartment_75_precentile))
    
    if left_compartment_75_precentile-left_compartment_25_precentile<right_compartment_75_precentile-right_compartment_25_precentile:
        model_probabilities_vector_without_nan = 1 - model_probabilities_vector_without_nan
        model_probabilities_vector_with_nan = 1 - model_probabilities_vector_with_nan

    plt.figure()  
    n_model, bins_model, patches_model = plt.hist(model_probabilities_vector_without_nan, 40)
    plt.show
    plt.savefig(local + str(directory) + '/Figures/model_probabilities_histogram_' + str(chrom) + '_' + str(experiment_resolution) + 'kb.png')
    plt.close()

    plt.figure()
    plt.plot(model_probabilities_vector_with_nan)
    plt.axis([0, np.size(model_probabilities_vector_with_nan), 0, 1])
    plt.savefig(local + str(directory) + '/Figures/model_probabilities_vector_with_nan_' + str(chrom) + '_' + str(experiment_resolution) + 'kb.png', dpi=1000)
    plt.close()  

main()

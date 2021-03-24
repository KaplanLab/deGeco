from functools import partial
import numpy as np

# This file is meant to be passed as --functions parameter to gc_model_main.
# It's used to fit a version of our model where the lambdas are created by a linear transformation on the
# subcompartments track, i.e. lambda = A * SC + B, for some (A, B) pair per subcompartment.

def hyper_scale_sc(subcompartments, non_nan_mask, number_of_states):
    assert number_of_states <= len(subcompartments)
    non_nan_count = non_nan_mask.sum()
    non_nan_sc = np.array(subcompartments)[:number_of_states, non_nan_mask]

    def scale_sc(params):
        scale_params = params.reshape((number_of_states, 2))
        lambdas = scale_params[:, 0, None] * non_nan_sc + scale_params[:, 1, None]

        return lambdas.T.flatten()

    param_count = 2 * number_of_states

    return dict(param_function=scale_sc, param_count=param_count)

chr19_subcompartments = np.load('chr19_subcompartments.npy')
chr19_valid_subcompartments = list(filter(np.any, chr19_subcompartments))

lambdas_hyper=partial(hyper_scale_sc, chr19_valid_subcompartments)

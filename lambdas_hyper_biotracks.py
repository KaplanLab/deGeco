from functools import partial
import autograd.numpy as np

# This file is meant to be passed as --functions parameter to gc_model_main.
# It's used to fit a version of our model where the lambdas are created using logistic regression of some biotracks, 
# i.e. lambda = exp(Tracks @ A + b)/sum(exp(Tracks * A_i + b_i)), for some (A_i, b_i) pair per state.

def hann(M, n=None):
    if n is None:
        n = M
    return 0.5 - 0.5 * np.cos(2*np.pi*np.arange(n)/(M-1))

def hyper_logistic_biotracks(biotracks, non_nan_mask, number_of_states, glm_func=np.exp, window=None, window_params=0):
    """
    biotracks should be of shape (n_bins, n_tracks)
    """
    non_nan_count = non_nan_mask.sum()
    non_nan_biotracks = np.array(biotracks)[non_nan_mask, :]
    biotracks_count = biotracks.shape[1]
    A_size = biotracks_count * number_of_states
    B_size = number_of_states
    _window = window if window is not None else lambda: np.array([1])
    A_start, B_start, W_start, _ = np.cumsum([0, A_size, B_size, window_params])

    def logistic_biotracks(params):
        A = params[A_start:B_start].reshape(biotracks_count, number_of_states)
        B = params[B_start:W_start]
        W = params[W_start:]
        window_instance = _window(*W)
        window_width = window_instance.size
        convolve = lambda x: np.convolve(x, window_instance, mode='same')
        windowed_biotracks = np.apply_along_axis(convolve, 0, non_nan_biotracks) / window_width
        lambdas = glm_func((A.T @ windowed_biotracks.T + B[:, None]).T).flatten()

        return lambdas.flatten()

    param_count = A_size + B_size + window_params
    bounds = [(-1, 1)] * param_count
    init_values = 2*np.random.random(param_count) -1

    return dict(param_function=logistic_biotracks, param_count=param_count, bounds=bounds, init_values=init_values)

# read biotracks and export if we're run from gc_model_main
if __name__ == '<run_path>':
    lambdas_hyper=partial(hyper_logistic_biotracks, get_biotracks())

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.math cimport log, exp
import numpy as np
from autograd.extend import primitive, defvjp

logp = None
grad_lambdas = None
grad_weights = None
cdef double grad_alpha
cdef double grad_beta


def dummy_vjp(ans, *args):
    def _vjp(g):
        return np.zeros(args[4].shape[0])
    return _vjp

def preallocate(logp_size, nbins, nstates):
    global logp
    global grad_lambdas
    global grad_weights

    logp = np.empty(logp_size, dtype=float)
    grad_weights = np.empty((nstates, nstates), dtype=float)
    grad_lambdas = np.empty((nbins, nstates), dtype=float)

def balance_counts(int[::1] bin1_id not None, int[::1] bin2_id not None, counts not None,
                   double[::1] balance_weights not None):
    balanced_counts = np.ndarray.astype(counts, float, copy=False)
    cdef double[::1] balanced_counts_view = balanced_counts
    cdef int i, j
    
    for row in range(balanced_counts_view.shape[0]):
        i = bin1_id[row]
        j = bin2_id[row]
        balanced_counts_view[row] *= balance_weights[i] * balance_weights[j]

    return balanced_counts

 # TODO: combine the different grad functions with the function the computes the value. Will allow us to do one
 # pass over the data instead of several, which is much more cache-friendly
def calc_logp_grad_lambdas(ans_py, lambdas_py not None, weights_py not None,
        double alpha, beta, bin1_id_py, bin2_id_py, chr_assoc_py, non_nan_map_py):
    def _vjp(g):
        cdef double[::1] ans = ans_py
        cdef double[:, ::1] lambdas = lambdas_py
        cdef double[:, ::1] weights = weights_py
        cdef int[::1] bin1_id = bin1_id_py
        cdef int[::1] bin2_id = bin2_id_py
        cdef long[::1] chr_assoc = chr_assoc_py
        cdef long[::1] non_nan_map = non_nan_map_py
        cdef Py_ssize_t bincount = bin1_id.shape[0]
        cdef Py_ssize_t nstates = weights.shape[0]
        global grad_lambdas
        global grad_weights
        global grad_alpha
        global grad_beta
        cdef double[:, ::1] grad_lambdas_view = grad_lambdas
        cdef double[:, ::1] grad_weights_view = grad_weights
        cdef int i, j, i_nn, j_nn, s1, s2
        cdef double gc, d_i, d_j, d_alpha, d_beta
        cdef double[:] g_view = g
        cdef int g_row = 0
        
        grad_lambdas_view[:, :] = 0
        grad_weights_view[:, :] = 0
        grad_alpha = 0
        grad_beta = 0

        for row in range(bincount):
            i = bin1_id[row]
            j = bin2_id[row]
            i_nn = non_nan_map[i]
            j_nn = non_nan_map[j]
            if i == j or i_nn < 0 or j_nn < 0:
                continue
            logdistance = log(j-i)
            dd = alpha * logdistance
            gc = exp(ans[g_row] - dd)
            for s1 in range(nstates):
                d_i = d_j = 0
                for s2 in range(nstates):
                    grad_weights_view[s1, s2] += g_view[g_row] * weights_jac_element(i_nn, j_nn, s1, s2, lambdas) / gc
                d_i, d_j = lambdas_jac_element(i_nn, j_nn, s1, lambdas, weights)
                d_i /= gc
                d_j /= gc
                grad_lambdas_view[i_nn, s1] += g_view[g_row] * d_i
                grad_lambdas_view[j_nn, s1] += g_view[g_row] * d_j

            d_alpha, d_beta = dd_jac_element(i, j, chr_assoc)
            grad_alpha += d_alpha * g_view[g_row]
            grad_beta += d_beta * g_view[g_row]
            g_row += 1
                
        return grad_lambdas
    return _vjp

def calc_logp_grad_weights(ans_py, lambdas_py not None, weights_py not None,
        alpha, beta, bin1_id_py, bin2_id_py, chr_assoc_py, non_nan_map_py):
    def _vjp(g):
        return grad_weights
    return _vjp

def calc_logp_grad_alpha(ans_py, lambdas_py not None, weights_py not None,
        alpha, beta, bin1_id_py, bin2_id_py, chr_assoc_py, non_nan_map_py):
    def _vjp(g):
        return grad_alpha
    return _vjp
    
def calc_logp_grad_beta(ans_py, lambdas_py not None, weights_py not None,
        alpha, beta, bin1_id_py, bin2_id_py, chr_assoc_py, non_nan_map_py):
    def _vjp(g):
        return grad_beta
    return _vjp

cdef inline double calc_dd(long i, long j, double alpha, double beta, long[::1] chr_assoc):
   # Check cis/trans
   if chr_assoc[i] == chr_assoc[j]:
       return alpha * log(j - i) # We iterate over the upper triangle, so i < j.
   else:
       return beta

cdef inline double calc_gc(long i, long j, double[:, ::1] lambdas, double[:, ::1] weights):
    cdef Py_ssize_t nstates = weights.shape[0]
    cdef double gc = 0
    cdef long s1, s2
    for s1 in range(nstates):
        for s2 in range(nstates):
            gc += lambdas[i, s1] * lambdas[j, s2] * weights[s1, s2]

    return gc

cdef inline (double, double) lambdas_jac_element(long i, long j, long s1, double[:, ::1] lambdas, double[:, ::1] weights):
    # This is _almost_ the jac element. The real jac element is devided by the full GC part, but since I sometimes need this
    # division and sometimes not, I preferred to add it when needed later
    cdef Py_ssize_t statecount = weights.shape[0]
    cdef double d_i, d_j
    cdef long s2
    
    d_i = d_j = 0
    for s2 in range(statecount):
        d_i += lambdas[j, s2] * weights[s1, s2]
        d_j += lambdas[i, s2] * weights[s1, s2]

    return d_i, d_j

cdef inline double weights_jac_element(long i, long j, long s1, long s2, double[:, ::1] lambdas):
    # This is _almost_ the jac element. The real jac element is devided by the full GC part, but since I sometimes need this
    # division and sometimes not, I preferred to add it when needed later
    return lambdas[i, s1] * lambdas[j, s2]
    
cdef inline (double, double) dd_jac_element(long i, long j, long[::1] chr_assoc):
    cdef double alpha_jac, beta_jac

    if chr_assoc[i] == chr_assoc[j]:
        alpha_jac = log(j - i)
        beta_jac = 0
    else:
        alpha_jac = 0
        beta_jac = 1

    return alpha_jac, beta_jac

@primitive
def calc_logp(double[:, ::1] lambdas not None, double[:, ::1] weights not None, double alpha, double beta,
        int[::1] bin1_id, int[::1] bin2_id, long[::1] chr_assoc, long[::1] non_nan_map):
    cdef Py_ssize_t bincount = bin1_id.shape[0]
    cdef double log_dd, gc
    cdef int i, j, i_nn, j_nn
    global logp
    out = logp
    cdef double[::1] out_view = out
    cdef int out_row = 0
    
    for row in range(bincount):
        i = bin1_id[row]
        j = bin2_id[row]
        i_nn = non_nan_map[i]
        j_nn = non_nan_map[j]
        if i == j or i_nn < 0 or j_nn < 0:
            continue
        log_dd = calc_dd(i, j, alpha, beta, chr_assoc)
        gc = calc_gc(i_nn, j_nn, lambdas, weights)
        out_view[out_row] = log_dd + log(gc)
        out_row += 1
        
    return out

defvjp(calc_logp,
       calc_logp_grad_lambdas,
       calc_logp_grad_weights,
       calc_logp_grad_alpha,
       calc_logp_grad_beta,
       dummy_vjp,
       dummy_vjp,
       dummy_vjp,
)

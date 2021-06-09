# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from cpython.exc cimport PyErr_CheckSignals
from libc.math cimport log, exp
import numpy as np
from autograd.extend import primitive, defvjp
from logsumexp cimport StreamingLogsumexp
from ssum cimport fsum_step

cdef double[:, ::1] grad_lambdas
cdef double[:, ::1] grad_weights
cdef double[:, ::1] grad_lambdas_part2
cdef double[:, ::1] grad_weights_part2
cdef double[:, ::1] grad_lambdas_c
cdef double[:, ::1] grad_weights_c
cdef double[:, ::1] grad_lambdas_part2_c
cdef double[:, ::1] grad_weights_part2_c
cdef double grad_alpha, grad_alpha_part2, grad_alpha_c, grad_alpha_part2_c
cdef double grad_beta, grad_beta_part2, grad_beta_c, grad_beta_part2_c

cpdef inline double calc_dd(long i, long j, double alpha, double beta, long[::1] chr_assoc):
   # Check cis/trans
   if chr_assoc[i] == chr_assoc[j]:
       return alpha * log(j - i) # We iterate over the upper triangle, so i < j.
   else:
       return beta

cpdef inline double calc_gc(long i, long j, double[:, ::1] lambdas, double[:, ::1] weights):
    cdef Py_ssize_t nstates = weights.shape[0]
    cdef double gc = 0
    cdef long s1, s2
    for s1 in range(nstates):
        for s2 in range(nstates):
            gc += lambdas[i, s1] * lambdas[j, s2] * weights[s1, s2]

    return gc

cpdef inline (double, double) lambdas_jac_element(long i, long j, long s1, double[:, ::1] lambdas, double[:, ::1] weights):
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

cpdef inline double weights_jac_element(long i, long j, long s1, long s2, double[:, ::1] lambdas):
    # This is _almost_ the jac element. The real jac element is devided by the full GC part, but since I sometimes need this
    # division and sometimes not, I preferred to add it when needed later
    return lambdas[i, s1] * lambdas[j, s2]
    
cpdef inline (double, double) dd_jac_element(long i, long j, long[::1] chr_assoc):
    cdef double alpha_jac, beta_jac

    if chr_assoc[i] == chr_assoc[j]:
        alpha_jac = log(j - i)
        beta_jac = 0
    else:
        alpha_jac = 0
        beta_jac = 1

    return alpha_jac, beta_jac

def preallocate(nbins, nstates):
    global grad_lambdas
    global grad_lambdas_part2
    global grad_weights
    global grad_weights_part2
    global grad_lambdas_c
    global grad_lambdas_part2_c
    global grad_weights_c
    global grad_weights_part2_c

    grad_weights = np.empty((nstates, nstates), dtype=float)
    grad_lambdas = np.empty((nbins, nstates), dtype=float)
    grad_weights_part2 = np.empty((nstates, nstates), dtype=float)
    grad_lambdas_part2 = np.empty((nbins, nstates), dtype=float)
    grad_weights_c = np.empty((nstates, nstates), dtype=float)
    grad_lambdas_c = np.empty((nbins, nstates), dtype=float)
    grad_weights_part2_c = np.empty((nstates, nstates), dtype=float)
    grad_lambdas_part2_c = np.empty((nbins, nstates), dtype=float)

cdef void grad_reset():
    global grad_lambdas
    global grad_weights
    global grad_alpha
    global grad_beta
    global grad_lambdas_part2
    global grad_weights_part2
    global grad_alpha_part2
    global grad_beta_part2
    global grad_alpha_c
    global grad_beta_c
    global grad_alpha_part2_c
    global grad_beta_part2_c
    global grad_lambdas_c
    global grad_lambdas_part2_c
    global grad_weights_c
    global grad_weights_part2_c

    grad_alpha = 0
    grad_beta = 0
    grad_alpha_part2 = 0
    grad_beta_part2 = 0
    grad_alpha_c = 0
    grad_beta_c = 0
    grad_alpha_part2_c = 0
    grad_beta_part2_c = 0
    grad_lambdas[:, :] = 0
    grad_lambdas_part2[:, :] = 0
    grad_weights[:, :] = 0
    grad_weights_part2[:, :] = 0
    grad_lambdas_c[:, :] = 0
    grad_lambdas_part2_c[:, :] = 0
    grad_weights_c[:, :] = 0
    grad_weights_part2_c[:, :] = 0

cdef void grad_update_gc(long i, long j, double gc, double dd, double x,
                        double[:, ::1] lambdas, double[:, ::1] weights, double amplification=1):
    global grad_lambdas
    global grad_weights
    global grad_lambdas_part2
    global grad_weights_part2
    global grad_lambdas_c
    global grad_weights_c
    global grad_lambdas_part2_c
    global grad_weights_part2_c

    cdef double d_i, d_j, d_w
    cdef long s1, s2
    cdef Py_ssize_t statecount = weights.shape[0]
    cdef double x_gc_ratio = x / gc

    for s1 in range(statecount):
        for s2 in range(statecount):
            d_w = weights_jac_element(i, j, s1, s2, lambdas)
            grad_weights[s1, s2], grad_weights_c[s1, s2] = fsum_step(grad_weights[s1, s2], grad_weights_c[s1, s2], d_w * x_gc_ratio)
            grad_weights_part2[s1, s2], grad_weights_part2_c[s1, s2] = fsum_step(grad_weights_part2[s1, s2], grad_weights_part2_c[s1, s2], d_w * dd * amplification)
        d_i, d_j = lambdas_jac_element(i, j, s1, lambdas, weights)
        grad_lambdas[i, s1], grad_lambdas_c[i, s1] = fsum_step(grad_lambdas[i, s1], grad_lambdas_c[i, s1], d_i * x_gc_ratio)
        grad_lambdas[j, s1], grad_lambdas_c[j, s1] = fsum_step(grad_lambdas[j, s1], grad_lambdas_c[j, s1], d_j * x_gc_ratio)
        grad_lambdas_part2[i, s1], grad_lambdas_part2_c[i, s1] = fsum_step(grad_lambdas_part2[i, s1], grad_lambdas_part2_c[i, s1], d_i * dd * amplification)
        grad_lambdas_part2[j, s1], grad_lambdas_part2_c[j, s1] = fsum_step(grad_lambdas_part2[j, s1], grad_lambdas_part2_c[j, s1], d_j * dd * amplification)

cdef void grad_update_dd(long i, long j, double p, double x, long[::1] chr_assoc, double amplification=1):
    cdef double alpha_jac, beta_jac
    global grad_alpha
    global grad_beta
    global grad_alpha_part2
    global grad_beta_part2
    global grad_alpha_c
    global grad_beta_c
    global grad_alpha_part2_c
    global grad_beta_part2_c

    alpha_jac, beta_jac = dd_jac_element(i, j, chr_assoc)
    grad_alpha, grad_alpha_c = fsum_step(grad_alpha, grad_alpha_c, alpha_jac * x)
    grad_beta, grad_beta_c = fsum_step(grad_beta, grad_beta_c, beta_jac * x)
    grad_alpha_part2, grad_alpha_part2_c = fsum_step(grad_alpha_part2, grad_alpha_part2_c, alpha_jac * p * amplification)
    grad_beta_part2, grad_beta_part2_c = fsum_step(grad_beta_part2, grad_beta_part2_c, beta_jac * p * amplification)

cdef void grad_finalize(double x_sum, double log_z):
    global grad_alpha
    global grad_beta
    global grad_alpha_part2
    global grad_beta_part2
    global grad_lambdas
    global grad_weights
    global grad_lambdas_part2
    global grad_weights_part2

    grad_ratio = x_sum * exp(-log_z)
    grad_lambdas -= grad_ratio * np.asarray(grad_lambdas_part2)
    grad_weights -= grad_ratio * np.asarray(grad_weights_part2)
    grad_alpha -= grad_ratio * grad_alpha_part2
    grad_beta -= grad_ratio * grad_beta_part2

def calc_likelihood_grad_lambdas(ans_py, lambdas not None, weights not None,
        alpha, beta, bin1_id, bin2_id, count, zero_indices, total_zero_count,
        chr_assoc, non_nan_map):
    def _vjp(g):
        return g * np.asarray(grad_lambdas)
    return _vjp

def calc_likelihood_grad_weights(ans_py, lambdas not None, weights not None,
        alpha, beta, bin1_id, bin2_id, count, zero_indices, total_zero_count,
        chr_assoc, non_nan_map):
    def _vjp(g):
        return g * np.asarray(grad_weights)
    return _vjp

def calc_likelihood_grad_alpha(ans_py, lambdas not None, weights not None,
        alpha, beta, bin1_id, bin2_id, count, zero_indices, total_zero_count,
        chr_assoc, non_nan_map):
    def _vjp(g):
        return g * grad_alpha
    return _vjp
    
def calc_likelihood_grad_beta(ans_py, lambdas not None, weights not None,
        alpha, beta, bin1_id, bin2_id, count, zero_indices, total_zero_count,
        chr_assoc, non_nan_map):
    def _vjp(g):
        return g * grad_beta
    return _vjp

@primitive
def calc_likelihood(double[:, ::1] lambdas not None, double[:, ::1] weights not None, double alpha, double beta,
        int[::1] bin1_id, int[::1] bin2_id, double[::1] count, int [:, ::1] zero_indices,
        long total_zero_count, long[::1] chr_assoc, long[::1] non_nan_map):

    cdef Py_ssize_t bincount = bin1_id.shape[0]
    cdef Py_ssize_t zerocount = zero_indices.shape[1] if zero_indices is not None else 0
    cdef double log_dd, gc, log_gc, logp, loglikelihood_part1=0, x_sum=0, log_z, x
    cdef double loglikelihood_part1_c=0, x_sum_c=0
    cdef int i, j, i_nn, j_nn
    
    cdef double log_amplification = log(total_zero_count) - log(zerocount)
    cdef double zero_amplification = exp(log_amplification)
    grad_reset()
    log_z_obj = StreamingLogsumexp()
    for row in range(bincount):
        i = bin1_id[row]
        j = bin2_id[row]
        i_nn = non_nan_map[i]
        j_nn = non_nan_map[j]
        if i == j or i_nn < 0 or j_nn < 0:
            continue
        x = count[row]
        log_dd = calc_dd(i, j, alpha, beta, chr_assoc)
        gc = calc_gc(i_nn, j_nn, lambdas, weights)
        log_gc = log(gc)
        logp = log_dd + log_gc
        loglikelihood_part1, loglikelihood_part1_c = fsum_step(loglikelihood_part1, loglikelihood_part1_c, x * logp)
        x_sum, x_sum_c = fsum_step(x_sum, x_sum_c, x) # TODO: Do this once, take as parameter?
        log_z_obj.update(logp)

        grad_update_gc(i_nn, j_nn, gc, exp(log_dd), x, lambdas, weights)
        grad_update_dd(i, j, exp(logp), x, chr_assoc)
    PyErr_CheckSignals()

    x = 0
    for row in range(zerocount):
        i = zero_indices[0, row]
        j = zero_indices[1, row]
        i_nn = non_nan_map[i]
        j_nn = non_nan_map[j]
        if i == j or i_nn < 0 or j_nn < 0:
            continue
        log_dd = calc_dd(i, j, alpha, beta, chr_assoc)
        gc = calc_gc(i_nn, j_nn, lambdas, weights)
        log_gc = log(gc)
        logp = log_dd + log_gc
        log_z_obj.update(logp + log_amplification)

        grad_update_gc(i_nn, j_nn, gc, exp(log_dd), x, lambdas, weights, zero_amplification)
        grad_update_dd(i, j, exp(logp), x, chr_assoc, zero_amplification)
    PyErr_CheckSignals()

    log_z = log_z_obj.result()
    grad_finalize(x_sum, log_z)
    PyErr_CheckSignals()

    return loglikelihood_part1 - x_sum * log_z

defvjp(calc_likelihood,
       calc_likelihood_grad_lambdas,
       calc_likelihood_grad_weights,
       calc_likelihood_grad_alpha,
       calc_likelihood_grad_beta,
)

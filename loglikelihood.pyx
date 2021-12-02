# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from cpython.exc cimport PyErr_CheckSignals
from libc.math cimport log, exp, isfinite
import numpy as np
from cython import parallel
from autograd.extend import primitive, defvjp
cimport logsumexp
from ssum cimport fsum_step
from libc.stdlib cimport malloc, free
cimport gap_sampler
import multiprocessing

cdef int total_threads
# dimensions: thread x bin x state x likelihood_component x fsum_component
cdef double[:, :, :, :, ::1] grad_lambdas
cdef double[:, :, :, :, ::1] grad_cis_weights
cdef double[:, :, :, :, ::1] grad_trans_weights
# dimensions: thread x likelihood_component x fsum_component
cdef double[:, :, ::1] grad_alpha
cdef double[:, :, ::1] grad_beta

cpdef (long, long) add_gap(Py_ssize_t bincount, long i, long j, int gap) nogil:
    cdef int new_j, new_i

    new_j = j + gap
    new_i = i
    while new_j >= bincount:
        new_i += 1
        new_j = new_j - bincount + new_i

    return new_i, new_j

cpdef inline double calc_dd(long i, long j, double alpha, double beta, long[::1] chr_assoc) nogil:
   # Check cis/trans
   if chr_assoc[i] == chr_assoc[j]:
       return alpha * log(j - i) # We iterate over the upper triangle, so i < j.
   else:
       return beta

cpdef inline double calc_gc(long i, long j, double[:, ::1] lambdas, double[:, ::1] weights) nogil:
    cdef Py_ssize_t nstates = weights.shape[0]
    cdef double gc = 0
    cdef long s1, s2
    for s1 in range(nstates):
        for s2 in range(nstates):
            gc += lambdas[i, s1] * lambdas[j, s2] * weights[s1, s2]

    return gc

cpdef inline (double, double) lambdas_jac_element(long i, long j, long s1, double[:, ::1] lambdas, double[:, ::1] weights) nogil:
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

cpdef inline double weights_jac_element(long i, long j, long s1, long s2, double[:, ::1] lambdas) nogil:
    # This is _almost_ the jac element. The real jac element is devided by the full GC part, but since I sometimes need this
    # division and sometimes not, I preferred to add it when needed later
    return lambdas[i, s1] * lambdas[j, s2]
    
cpdef inline (double, double) dd_jac_element(long i, long j, long[::1] chr_assoc) nogil:
    cdef double alpha_jac, beta_jac

    if chr_assoc[i] == chr_assoc[j]:
        alpha_jac = log(j - i)
        beta_jac = 0
    else:
        alpha_jac = 0
        beta_jac = 1

    return alpha_jac, beta_jac

def preallocate(nbins, nstates, nthreads=1):
    global grad_lambdas
    global grad_cis_weights
    global grad_trans_weights
    global grad_alpha
    global grad_beta
    global total_threads

    if not nthreads:
        nthreads = multiprocessing.cpu_count()
    total_threads = nthreads

    grad_cis_weights = np.empty((nthreads, nstates, nstates, 2, 2), dtype=float)
    grad_trans_weights = np.empty((nthreads, nstates, nstates, 2, 2), dtype=float)
    grad_lambdas = np.empty((nthreads, nbins, nstates, 2, 2), dtype=float)
    grad_alpha = np.empty((nthreads, 2, 2), dtype=float)
    grad_beta = np.empty((nthreads, 2, 2), dtype=float)

cdef void grad_reset():
    global grad_lambdas
    global grad_cis_weights
    global grad_trans_weights
    global grad_alpha
    global grad_beta

    grad_alpha[:] = 0
    grad_beta[:] = 0
    grad_lambdas[:] = 0
    grad_cis_weights[:] = 0
    grad_trans_weights[:] = 0

cdef void grad_update_gc(int thread, long i, long j, double gc, double dd, double x,
                        double[:, ::1] lambdas, double[:, ::1] weights, int cis, double amplification=1) nogil:
    global grad_lambdas
    global grad_cis_weights
    global grad_trans_weights

    cdef double d_i, d_j, d_w
    cdef long s1, s2
    cdef Py_ssize_t statecount = weights.shape[0]
    cdef double x_gc_ratio = x / gc
    cdef double[:, :, :, :, ::1] grad_weights

    if cis:
        grad_weights = grad_cis_weights
    else:
        grad_weights = grad_trans_weights
    for s1 in range(statecount):
        for s2 in range(statecount):
            d_w = weights_jac_element(i, j, s1, s2, lambdas)
            grad_weights[thread, s1, s2, 0, 0], grad_weights[thread, s1, s2, 0, 1] = fsum_step(grad_weights[thread, s1, s2, 0, 0], grad_weights[thread, s1, s2, 0, 1], d_w * x_gc_ratio)
            grad_weights[thread, s1, s2, 1, 0], grad_weights[thread, s1, s2, 1, 1] = fsum_step(grad_weights[thread, s1, s2, 1, 0], grad_weights[thread, s1, s2, 1, 1], d_w * dd * amplification)
        d_i, d_j = lambdas_jac_element(i, j, s1, lambdas, weights)
        grad_lambdas[thread, i, s1, 0, 0], grad_lambdas[thread, i, s1, 0, 1] = fsum_step(grad_lambdas[thread, i, s1, 0, 0], grad_lambdas[thread, i, s1, 0, 1], d_i * x_gc_ratio)
        grad_lambdas[thread, i, s1, 1, 0], grad_lambdas[thread, i, s1, 1, 1] = fsum_step(grad_lambdas[thread, i, s1, 1, 0], grad_lambdas[thread, i, s1, 1, 1], d_i * dd * amplification)
        grad_lambdas[thread, j, s1, 0, 0], grad_lambdas[thread, j, s1, 0, 1] = fsum_step(grad_lambdas[thread, j, s1, 0, 0], grad_lambdas[thread, j, s1, 0, 1], d_j * x_gc_ratio)
        grad_lambdas[thread, j, s1, 1, 0], grad_lambdas[thread, j, s1, 1, 1] = fsum_step(grad_lambdas[thread, j, s1, 1, 0], grad_lambdas[thread, j, s1, 1, 1], d_j * dd * amplification)

cdef void grad_update_dd(int thread, long i, long j, double p, double x, long[::1] chr_assoc, double amplification=1) nogil:
    cdef double alpha_jac, beta_jac
    global grad_alpha
    global grad_beta

    alpha_jac, beta_jac = dd_jac_element(i, j, chr_assoc)
    grad_alpha[thread, 0, 0], grad_alpha[thread, 0, 1] = fsum_step(grad_alpha[thread, 0, 0], grad_alpha[thread, 0, 1], alpha_jac * x)
    grad_alpha[thread, 1, 0], grad_alpha[thread, 1, 1] = fsum_step(grad_alpha[thread, 1, 0], grad_alpha[thread, 1, 1], alpha_jac * p * amplification)
    grad_beta[thread, 0, 0], grad_beta[thread, 0, 1] = fsum_step(grad_beta[thread, 0, 0], grad_beta[thread, 0, 1], beta_jac * x)
    grad_beta[thread, 1, 0], grad_beta[thread, 1, 1] = fsum_step(grad_beta[thread, 1, 0], grad_beta[thread, 1, 1], beta_jac * p * amplification)

cdef void grad_finalize(double x_sum, double log_z, int total_threads):
    global grad_lambdas
    global grad_cis_weights
    global grad_trans_weights
    global grad_alpha
    global grad_beta

    cdef Py_ssize_t t, i, j
    cdef double part1_part2_diff

    grad_ratio = x_sum * exp(-log_z)
    # Combine part1 and part2 of all gradients and sum results from all threads to array of first thread
    for t in range(total_threads):
        part1_part2_diff = grad_alpha[t, 0, 0] - grad_ratio * grad_alpha[t, 1, 0]
        if t == 0:
            grad_alpha[0, 0, 0] = part1_part2_diff
        else:
            grad_alpha[0, 0, 0] += part1_part2_diff
        part1_part2_diff = grad_beta[t, 0, 0] - grad_ratio * grad_beta[t, 1, 0]
        if t == 0:
            grad_beta[0, 0, 0] = part1_part2_diff
        else:
            grad_beta[0, 0, 0] += part1_part2_diff

        for i in range(grad_lambdas.shape[1]):
            for j in range(grad_lambdas.shape[2]):
                part1_part2_diff = grad_lambdas[t, i, j, 0, 0] - grad_ratio * grad_lambdas[t, i, j, 1, 0]
                if t == 0:
                    grad_lambdas[0, i, j, 0, 0] = part1_part2_diff
                else:
                    grad_lambdas[0, i, j, 0, 0] += part1_part2_diff
        for i in range(grad_cis_weights.shape[1]):
            for j in range(grad_cis_weights.shape[2]):
                part1_part2_diff = grad_cis_weights[t, i, j, 0, 0] - grad_ratio * np.asarray(grad_cis_weights[t, i, j, 1, 0])
                if t == 0:
                    grad_cis_weights[0, i, j, 0, 0] = part1_part2_diff
                else:
                    grad_cis_weights[0, i, j, 0, 0] += part1_part2_diff
                part1_part2_diff = grad_trans_weights[t, i, j, 0, 0] - grad_ratio * np.asarray(grad_trans_weights[t, i, j, 1, 0])
                if t == 0:
                    grad_trans_weights[0, i, j, 0, 0] = part1_part2_diff
                else:
                    grad_trans_weights[0, i, j, 0, 0] += part1_part2_diff

def calc_likelihood_grad_lambdas(ans_py, lambdas not None, cis_weights not None, trans_weights not None,
        alpha, beta, bin1_id, bin2_id, count, zero_indices, total_zero_count, chr_assoc, non_nan_map, zeros_start=0, zeros_step=1):
    def _vjp(g):
        return g * np.asarray(grad_lambdas[0, :, :, 0, 0])
    return _vjp

def calc_likelihood_grad_cis_weights(ans_py, lambdas not None, cis_weights not None, trans_weights not None,
        alpha, beta, bin1_id, bin2_id, count, zero_indices, total_zero_count, chr_assoc, non_nan_map, zeros_start=0, zeros_step=1):
    def _vjp(g):
        return g * np.asarray(grad_cis_weights[0, :, :, 0, 0])
    return _vjp

def calc_likelihood_grad_trans_weights(ans_py, lambdas not None, cis_weights not None, trans_weights not None,
        alpha, beta, bin1_id, bin2_id, count, zero_indices, total_zero_count, chr_assoc, non_nan_map, zeros_start=0, zeros_step=1):
    def _vjp(g):
        return g * np.asarray(grad_trans_weights[0, :, :, 0, 0])
    return _vjp

def calc_likelihood_grad_alpha(ans_py, lambdas not None, cis_weights not None, trans_weights not None,
        alpha, beta, bin1_id, bin2_id, count, zero_indices, total_zero_count, chr_assoc, non_nan_map, zeros_start=0, zeros_step=1):
    def _vjp(g):
        return g * grad_alpha[0, 0, 0]
    return _vjp
    
def calc_likelihood_grad_beta(ans_py, lambdas not None, cis_weights not None, trans_weights not None,
        alpha, beta, bin1_id, bin2_id, count, zero_indices, total_zero_count, chr_assoc, non_nan_map, zeros_start=0, zeros_step=1):
    def _vjp(g):
        return g * grad_beta[0, 0, 0]
    return _vjp

@primitive
def calc_likelihood(double[:, ::1] lambdas not None, double[:, ::1] cis_weights not None,
        double[:, ::1] trans_weights not None, double alpha, double beta, int[::1] bin1_id, int[::1] bin2_id,
        double[::1] count, long [::1] zero_indices, long total_zero_count, long[::1] chr_assoc,
        long[::1] non_nan_map, int zeros_start=0, int zeros_step=1):
    cdef Py_ssize_t bincount = bin1_id.shape[0]
    cdef Py_ssize_t row1, row2
    cdef double log_z, log_z_local, x1, x2=0
    cdef double loglikelihood_part1 = 0, x_sum = 0
    cdef double[::1] reduction_res = np.zeros(2)
    cdef double *ll_local
    cdef double *x_sum_local
    cdef double log_dd1, gc1, log_gc1, logp1
    cdef double log_dd2, gc2, log_gc2, logp2
    cdef int i1, j1, i_nn1, j_nn1
    cdef int i2, j2, i_nn2, j_nn2
    cdef int thread_num
    cdef logsumexp.lse log_z_obj
    cdef logsumexp.lse* log_z_obj_local
    cdef double zerocount = total_zero_count / zeros_step
    cdef double log_zero_amplification = log(total_zero_count / zerocount)
    cdef long[::1] pos = np.zeros(total_threads, dtype=int)
    cdef int i_gap, row, col
    cdef long nbins = non_nan_map.shape[0]
    grad_reset()
    logsumexp.lse_init(&log_z_obj)
    with nogil, parallel.parallel(num_threads=total_threads):
        ll_local = <double*>malloc(sizeof(double)*2)
        x_sum_local = <double*>malloc(sizeof(double)*2)
        ll_local[0] = ll_local[1] = 0
        x_sum_local[0] = x_sum_local[1] = 0
        thread_num = parallel.threadid()
        log_z_obj_local = <logsumexp.lse*>malloc(sizeof(logsumexp.lse))
        logsumexp.lse_init(log_z_obj_local)
        for row1 in parallel.prange(bincount):
            i1 = bin1_id[row1]
            j1 = bin2_id[row1]
            i_nn1 = non_nan_map[i1]
            j_nn1 = non_nan_map[j1]
            if i1 == j1 or i_nn1 < 0 or j_nn1 < 0:
                continue
            x1 = count[row1]
            log_dd1 = calc_dd(i1, j1, alpha, beta, chr_assoc)
            if chr_assoc[i1] == chr_assoc[j1]:
                gc1 = calc_gc(i_nn1, j_nn1, lambdas, cis_weights)
                grad_update_gc(thread_num, i_nn1, j_nn1, gc1, exp(log_dd1), x1, lambdas, cis_weights, cis=True)
            else:
                gc1 = calc_gc(i_nn1, j_nn1, lambdas, trans_weights)
                grad_update_gc(thread_num, i_nn1, j_nn1, gc1, exp(log_dd1), x1, lambdas, trans_weights, cis=False)
            log_gc1 = log(gc1)
            logp1 = log_dd1 + log_gc1
            ll_local[0], ll_local[1] = fsum_step(ll_local[0], ll_local[1], x1 * logp1)
            x_sum_local[0], x_sum_local[1] = fsum_step(x_sum_local[0], x_sum_local[1], x1) # TODO: Do this once, take as parameter?
            logsumexp.lse_update(log_z_obj_local, logp1)

            grad_update_dd(thread_num, i1, j1, exp(logp1), x1, chr_assoc)
        with gil:
            PyErr_CheckSignals()
            reduction_res[0] += ll_local[0]
            reduction_res[1] += x_sum_local[0]

        pos[thread_num] = 0
        for row2 in parallel.prange(zeros_start, total_zero_count, zeros_step):
            pos[thread_num] = gap_sampler.get_position(row2, zero_indices, pos[thread_num])
            i_gap = gap_sampler.get_gap(zero_indices, pos[thread_num], row2)
            row, col = gap_sampler.pos2rowcol(pos[thread_num], bin1_id, bin2_id)
            i2, j2 = gap_sampler.move_right(nbins, row, col, i_gap)
            i_nn2 = non_nan_map[i2]
            j_nn2 = non_nan_map[j2]
            if i2 == j2 or i_nn2 < 0 or j_nn2 < 0:
                continue
            log_dd2 = calc_dd(i2, j2, alpha, beta, chr_assoc)
            if chr_assoc[i2] == chr_assoc[j2]:
                gc2 = calc_gc(i_nn2, j_nn2, lambdas, cis_weights)
                grad_update_gc(thread_num, i_nn2, j_nn2, gc2, exp(log_dd2), x2, lambdas, cis_weights, cis=True)
            else:
                gc2 = calc_gc(i_nn2, j_nn2, lambdas, trans_weights)
                grad_update_gc(thread_num, i_nn2, j_nn2, gc2, exp(log_dd2), x2, lambdas, trans_weights, cis=False)
            log_gc2 = log(gc2)
            logp2 = log_dd2 + log_gc2
            logsumexp.lse_update(log_z_obj_local, logp2)

            grad_update_dd(thread_num, i2, j2, exp(logp2), x2, chr_assoc)
        with gil:
            PyErr_CheckSignals()
            log_z_local = logsumexp.lse_result(log_z_obj_local)
            if isfinite(log_z_local):
                logsumexp.lse_update(&log_z_obj, log_z_local)
        free(ll_local)
        free(x_sum_local)
        free(log_z_obj_local)

    loglikelihood_part1, x_sum = reduction_res
    log_z = logsumexp.lse_result(&log_z_obj)
    grad_finalize(x_sum, log_z, total_threads)
    PyErr_CheckSignals()

    return loglikelihood_part1 - x_sum * log_z

defvjp(calc_likelihood,
       calc_likelihood_grad_lambdas,
       calc_likelihood_grad_cis_weights,
       calc_likelihood_grad_trans_weights,
       calc_likelihood_grad_alpha,
       calc_likelihood_grad_beta,
)

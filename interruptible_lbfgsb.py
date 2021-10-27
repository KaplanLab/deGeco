## License for the Python wrapper
## ==============================

## Copyright (c) 2004 David M. Cooke <cookedm@physics.mcmaster.ca>

## Permission is hereby granted, free of charge, to any person obtaining a
## copy of this software and associated documentation files (the "Software"),
## to deal in the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:

## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.

## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS IN THE SOFTWARE.

## Modifications by Travis Oliphant and Enthought, Inc. for inclusion in SciPy
## Modifications by Hagai Kariti for making the function interruptible

import warnings
import numpy as np
from numpy import array, asarray, int32, float64, zeros
from scipy.optimize import _lbfgsb
from scipy.optimize.optimize import (OptimizeResult, OptimizeWarning)
from scipy.optimize._constraints import old_bound_to_new

from scipy.sparse.linalg import LinearOperator
from scipy.optimize.lbfgsb import LbfgsInvHessProduct

__all__ = ['minimize']

def wrap_function(function, args):
   ncalls = [0]
   if function is None:
       return ncalls, None

   def function_wrapper(*wrapper_args):
       ncalls[0] += 1
       return function(*wrapper_args, *args)

   return ncalls, function_wrapper

def check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        # Stack level 4: this is called from _minimize_*, which is
        # called from another function in SciPy. Level 4 is the first
        # level in user code.
        warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, 4)
 
def minimize(fun, x0, args=(), jac=None, bounds=None,
             disp=None, maxcor=10, ftol=2.2204460492503131e-09,
             gtol=1e-5, eps=1e-8, maxfun=15000, maxiter=15000,
             iprint=-1, callback=None, maxls=20,
             finite_diff_rel_step=None, internal_state=None,
             # These options are passed to all custom optimizers, we ignore them
             hess=None, hessp=None, constraints=None, **unknown_options):
    """
    Minimize a scalar function of one or more variables using the L-BFGS-B
    algorithm.

    Options
    -------
    disp : None or int
        If `disp is None` (the default), then the supplied version of `iprint`
        is used. If `disp is not None`, then it overrides the supplied version
        of `iprint` with the behaviour you outlined.
    maxcor : int
        The maximum number of variable metric corrections used to
        define the limited memory matrix. (The limited memory BFGS
        method does not store the full hessian but uses this many terms
        in an approximation to it.)
    ftol : float
        The iteration stops when ``(f^k -
        f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
    gtol : float
        The iteration will stop when ``max{|proj g_i | i = 1, ..., n}
        <= gtol`` where ``pg_i`` is the i-th component of the
        projected gradient.
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    maxfun : int
        Maximum number of function evaluations.
    maxiter : int
        Maximum number of iterations.
    iprint : int, optional
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint = 0``    print only one line at the last iteration;
        ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;
        ``iprint = 99``   print details of every iteration except n-vectors;
        ``iprint = 100``  print also the changes of active set and final x;
        ``iprint > 100``  print details of every iteration including x and g.
    callback : callable, optional
        Called after each iteration, as ``callback(xk, st)``, where ``xk`` is
        the current parameter vector and ``st`` is a tuple of the internal
        state of the algorithm.
    maxls : int, optional
        Maximum number of line search steps (per iteration). Default is 20.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    internal_state: None or tuple, optional
        Pass the initial internal state of the algorithm, to resume after an
        interruption.

    Notes
    -----
    The option `ftol` is exposed via the `scipy.optimize.minimize` interface,
    but calling `scipy.optimize.fmin_l_bfgs_b` directly exposes `factr`. The
    relationship between the two is ``ftol = factr * numpy.finfo(float).eps``.
    I.e., `factr` multiplies the default machine floating-point precision to
    arrive at `ftol`.

    """
    check_unknown_options(unknown_options)
    m = maxcor
    pgtol = gtol
    factr = ftol / np.finfo(float).eps

    x0 = asarray(x0).ravel()
    n, = x0.shape

    if bounds is None:
        bounds = [(None, None)] * n
    if len(bounds) != n:
        raise ValueError('length of x0 != length of bounds')

    # unbounded variables must use None, not +-inf, for optimizer to work properly
    bounds = [(None if l == -np.inf else l, None if u == np.inf else u) for l, u in bounds]
    # LBFGSB is sent 'old-style' bounds, 'new-style' bounds are required by
    # approx_derivative and ScalarFunction
    new_bounds = old_bound_to_new(bounds)

    # check bounds
    if (new_bounds[0] > new_bounds[1]).any():
        raise ValueError("LBFGSB - one of the lower bounds is greater than an upper bound.")

    # initial vector must lie within the bounds. Otherwise ScalarFunction and
    # approx_derivative will cause problems
    x0 = np.clip(x0, new_bounds[0], new_bounds[1])

    if disp is not None:
        if disp == 0:
            iprint = -1
        else:
            iprint = disp

    n_function_evals, fun = wrap_function(fun, ())
    if jac is None:
        raise NotImplementedError
    else:
        def func_and_grad(x):
            f = fun(x, *args)
            g = jac(x, *args)
            return f, g

    try:
        fortran_int = _lbfgsb.types.intvar.dtype
    except AttributeError:
        # Older scipy versions supported only int32
        fortran_int = int32

    nbd = zeros(n, fortran_int)
    low_bnd = zeros(n, float64)
    upper_bnd = zeros(n, float64)
    bounds_map = {(None, None): 0,
                  (1, None): 1,
                  (1, 1): 2,
                  (None, 1): 3}
    for i in range(0, n):
        l, u = bounds[i]
        if l is not None:
            low_bnd[i] = l
            l = 1
        if u is not None:
            upper_bnd[i] = u
            u = 1
        nbd[i] = bounds_map[l, u]

    if not maxls > 0:
        raise ValueError('maxls must be positive.')

    if internal_state is None:
        x = array(x0, float64)
        f = array(0.0, float64)
        g = zeros((n,), float64)
        wa = zeros(2*m*n + 5*n + 11*m*m + 8*m, float64)
        iwa = zeros(3*n, fortran_int)
        task = zeros(1, 'S60')
        csave = zeros(1, 'S60')
        lsave = zeros(4, fortran_int)
        isave = zeros(44, fortran_int)
        dsave = zeros(29, float64)

        task[:] = 'START'

        n_iterations = 0
    else:
        x, f, g, wa, iwa, task, csave, lsave, isave, dsave, n_iterations = internal_state

    while 1:
        # x, f, g, wa, iwa, task, csave, lsave, isave, dsave = \
        _lbfgsb.setulb(m, x, low_bnd, upper_bnd, nbd, f, g, factr,
                       pgtol, wa, iwa, task, iprint, csave, lsave,
                       isave, dsave, maxls)
        task_str = task.tobytes()
        if task_str.startswith(b'FG'):
            # The minimization routine wants f and g at the current x.
            # Note that interruptions due to maxfun are postponed
            # until the completion of the current minimization iteration.
            # Overwrite f and g:
            f, g = func_and_grad(x)
        elif task_str.startswith(b'NEW_X'):
            # new iteration
            n_iterations += 1
            if callback is not None:
                callback(x, f, g, wa, iwa, task, csave, lsave, isave, dsave, n_iterations)

            if n_iterations >= maxiter:
                task[:] = 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
            elif n_function_evals[0] > maxfun:
                task[:] = ('STOP: TOTAL NO. of f AND g EVALUATIONS '
                           'EXCEEDS LIMIT')
        else:
            break

    task_str = task.tobytes().strip(b'\x00').strip()
    if task_str.startswith(b'CONV'):
        warnflag = 0
    elif n_function_evals[0] > maxfun or n_iterations >= maxiter:
        warnflag = 1
    else:
        warnflag = 2

    # These two portions of the workspace are described in the mainlb
    # subroutine in lbfgsb.f. See line 363.
    s = wa[0: m*n].reshape(m, n)
    y = wa[m*n: 2*m*n].reshape(m, n)

    # See lbfgsb.f line 160 for this portion of the workspace.
    # isave(31) = the total number of BFGS updates prior the current iteration;
    n_bfgs_updates = isave[30]

    n_corrs = min(n_bfgs_updates, maxcor)
    hess_inv = LbfgsInvHessProduct(s[:n_corrs], y[:n_corrs])

    task_str = task_str.decode()
    return OptimizeResult(fun=f, jac=g, nfev=n_function_evals[0],
                          njev=n_function_evals[0],
                          nit=n_iterations, status=warnflag, message=task_str,
                          x=x, success=(warnflag == 0), hess_inv=hess_inv)

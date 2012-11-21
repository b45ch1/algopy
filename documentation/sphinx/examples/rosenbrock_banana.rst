Minimization of the Rosenbrock Function
-----------------------------------------------

In this example we want to use AlgoPy to help compute the
minimum of the non-convex bivariate
`Rosenbrock function <http://en.wikipedia.org/wiki/Rosenbrock_function>`_

.. math::
    f(x, y) = (1 - x)^2 + 100 (y - x^2)^2

The idea is that by using AlgoPy to provide the gradient and hessian
of the objective function,
the nonlinear optimization procedures in
`scipy.optimize <http://docs.scipy.org/doc/scipy/reference/optimize.html>`_
will more easily find the :math:`x` and :math:`y` values
that minimize :math:`f(x, y)`.
Here is the python code:

.. literalinclude:: rosenbrock_banana.py


Here is its output::

    Try to find the minimum of the Rosenbrock banana function.
    This is at f(1, 1) = 0 but the function is a bit tricky.
    To make the search difficult we will start far from the min.

    initial guess:
    [-1.2  1. ]

    strategy: default (Nelder-Mead)
    options: default
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 85
             Function evaluations: 159
    [ 1.00002202  1.00004222]

    strategy: ncg
    options: default
    gradient: autodiff
    hessian: autodiff
    Optimization terminated successfully.
             Current function value: 3.811010
             Iterations: 39
             Function evaluations: 41
             Gradient evaluations: 39
             Hessian evaluations: 39
    [-0.95155681  0.91039596]

    strategy: ncg
    options: default
    gradient: autodiff
    hessian: finite differences
    Optimization terminated successfully.
             Current function value: 3.810996
             Iterations: 39
             Function evaluations: 41
             Gradient evaluations: 185
             Hessian evaluations: 0
    [-0.95155309  0.91038895]

    strategy: bfgs
    options: default
    gradient: autodiff
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 31
             Function evaluations: 45
             Gradient evaluations: 45
    [ 0.99999933  0.99999865]

    strategy: bfgs
    options: default
    gradient: finite differences
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 31
             Function evaluations: 180
             Gradient evaluations: 45
    [ 0.99999486  0.9999897 ]

    strategy: slsqp
    options: default
    gradient: autodiff
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: 1.12238027858e-08
                Iterations: 34
                Function evaluations: 47
                Gradient evaluations: 34
    [ 0.99992192  0.99985101]

    strategy: slsqp
    options: default
    gradient: finite differences
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: 1.20063082621e-08
                Iterations: 34
                Function evaluations: 149
                Gradient evaluations: 34
    [ 0.99991762  0.99984247]

    strategy: powell
    options: default
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 23
             Function evaluations: 665
    [ 1.  1.]


The best way to find the minimum of this function is not so clear.
One confounding factor is that the various search strategies do not
necessarily have comparable default stopping criteria.
Another factor is that the direct function evaluation,
the AlgoPy gradient evaluation, and the finite differences
gradient approximation may each have different evaluation speeds.

On one hand the Newton conjugate gradient search
fails to find the right minimum despite its wealth of
AlgoPy-provided information about the objective function,
whereas the Powell search finds the minimum using only
direct evaluation of the objective function.
On the other hand we see that the BFGS search,
which succeeds in finding the minimum,
has improved accuracy when it uses AlgoPy to compute the gradient
as opposed to computing the gradient by finite differences.

Although not implemented here,
perhaps the nonlinear optimization search strategies available in
`IPOPT <https://projects.coin-or.org/Ipopt>`_
would make better use of the gradient and hessian,
as suggested by
`this vignette <http://www.ucl.ac.uk/~uctpjyy/downloads/ipoptr.pdf>`_
for the
`R interface <http://www.ucl.ac.uk/~uctpjyy/ipoptr.html>`_
to IPOPT.
This comparison has not yet been made using AlgoPy,
because the Python interface to IPOPT is more complicated to set up and use
than the scipy.optimize procedures.


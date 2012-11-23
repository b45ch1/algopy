Minimization of the Himmelblau Function
-----------------------------------------------

In this example we want to use AlgoPy to help compute the
minimum of the non-convex multi-modal bivariate
`Himmelblau function <http://en.wikipedia.org/wiki/Himmelblau's_function>`_

.. math::
    f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

The idea is that by using AlgoPy to provide the gradient and hessian
of the objective function,
the nonlinear optimization procedures in
`scipy.optimize <http://docs.scipy.org/doc/scipy/reference/optimize.html>`_
will more easily find the :math:`x` and :math:`y` values
that minimize :math:`f(x, y)`.
Here is the python code:

.. literalinclude:: himmelblau_minimization.py


Here is its output::

    properties of the function at a local min:
    point:
    [ 3.  2.]
    function value:
    0.0
    autodiff gradient:
    [ 0.  0.]
    finite differences gradient:
    [ 0.  0.]
    autodiff hessian:
    [[ 74.  20.]
     [ 20.  34.]]
    finite differences hessian:
    [[ 74.  20.]
     [ 20.  34.]]

    ---------------------------------------------------------
    searches beginning from the easier init point [ 3.1  2.1]
    ---------------------------------------------------------

    properties of the function at the initial guess:
    point:
    [ 3.1  2.1]
    function value:
    0.7642
    autodiff gradient:
    [ 9.824  5.704]
    finite differences gradient:
    [ 9.824  5.704]
    autodiff hessian:
    [[ 81.72  20.8 ]
     [ 20.8   39.32]]
    finite differences hessian:
    [[ 81.72  20.8 ]
     [ 20.8   39.32]]

    strategy: default (Nelder-Mead)
    options: default
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 31
             Function evaluations: 59
    [ 2.99997347  2.0000045 ]

    strategy: ncg
    options: default
    gradient: autodiff
    hessian: autodiff
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 4
             Function evaluations: 5
             Gradient evaluations: 4
             Hessian evaluations: 4
    [ 3.  2.]

    strategy: ncg
    options: default
    gradient: autodiff
    hessian: finite differences
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 4
             Function evaluations: 5
             Gradient evaluations: 18
             Hessian evaluations: 0
    [ 3.  2.]

    strategy: bfgs
    options: default
    gradient: autodiff
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 5
             Function evaluations: 9
             Gradient evaluations: 9
    [ 2.99999993  1.99999986]

    strategy: bfgs
    options: default
    gradient: finite differences
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 5
             Function evaluations: 36
             Gradient evaluations: 9
    [ 2.99999993  1.99999986]

    strategy: slsqp
    options: default
    gradient: autodiff
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: 2.68119275432e-08
                Iterations: 5
                Function evaluations: 9
                Gradient evaluations: 5
    [ 2.99997095  2.00001136]

    strategy: slsqp
    options: default
    gradient: finite differences
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: 2.68250888103e-08
                Iterations: 5
                Function evaluations: 24
                Gradient evaluations: 5
    [ 2.99997095  2.00001136]

    strategy: powell
    options: default
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 5
             Function evaluations: 126
    [ 3.  2.]

    strategy: tnc
    options: default
    gradient: autodiff
    (array([ 2.999999,  2.000002]), 11, 1)

    strategy: tnc
    options: default
    gradient: finite differences
    (array([ 2.99999646,  2.00000762]), 14, 1)


    ---------------------------------------------------------
    searches beginning from the more difficult init point [-0.27 -0.9 ]
    ---------------------------------------------------------

    properties of the function at the initial guess:
    point:
    [-0.27 -0.9 ]
    function value:
    181.61189441
    autodiff gradient:
    [-0.146732 -0.3982  ]
    finite differences gradient:
    [-0.146732 -0.3982  ]
    autodiff hessian:
    [[-44.7252  -4.68  ]
     [ -4.68   -17.36  ]]
    finite differences hessian:
    [[-44.7252  -4.68  ]
     [ -4.68   -17.36  ]]

    strategy: default (Nelder-Mead)
    options: default
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 54
             Function evaluations: 105
    [-2.80514623  3.13132056]

    strategy: ncg
    options: default
    gradient: autodiff
    hessian: autodiff
    Optimization terminated successfully.
             Current function value: 181.611894
             Iterations: 1
             Function evaluations: 5
             Gradient evaluations: 1
             Hessian evaluations: 1
    [-0.26999996 -0.89999989]

    strategy: ncg
    options: default
    gradient: autodiff
    hessian: finite differences
    Optimization terminated successfully.
             Current function value: 181.611894
             Iterations: 1
             Function evaluations: 5
             Gradient evaluations: 3
             Hessian evaluations: 0
    [-0.26999996 -0.89999989]

    strategy: bfgs
    options: default
    gradient: autodiff
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 11
             Function evaluations: 29
             Gradient evaluations: 29
    [ 3.          1.99999999]

    strategy: bfgs
    options: default
    gradient: finite differences
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 11
             Function evaluations: 116
             Gradient evaluations: 29
    [ 3.          1.99999999]

    strategy: slsqp
    options: default
    gradient: autodiff
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: 6.19262349912e-09
                Iterations: 10
                Function evaluations: 22
                Gradient evaluations: 10
    [ 2.99999684  2.00002046]

    strategy: slsqp
    options: default
    gradient: finite differences
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: 6.18718154108e-09
                Iterations: 10
                Function evaluations: 52
                Gradient evaluations: 10
    [ 2.99999683  2.00002045]

    strategy: powell
    options: default
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 6
             Function evaluations: 155
    [ 3.58442834 -1.84812653]

    strategy: tnc
    options: default
    gradient: autodiff
    (array([ 3.,  2.]), 42, 1)

    strategy: tnc
    options: default
    gradient: finite differences
    (array([ 2.99999981,  1.99999997]), 39, 1)




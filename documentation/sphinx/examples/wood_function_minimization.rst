Minimization of Wood's Function
-----------------------------------------------

In this example we want to use AlgoPy to help compute the
minimum of Wood's function.
This is from Problem 3.1 of 
`A truncated Newton method
with nonmonotone line search for unconstrained optimization
<http://dx.doi.org/10.1007/BF00940345>`_
by Grippo et al. 1989.

.. math::
    f(x_1, x_2, x_3, x_4)
        = & 100(x_1^2 - x_2)^2 + (x_1-1)^2 + (x_3-1)^2 \\
          & + 90(x_3^2-x_4)^2 \\
          & + 10.1 \left( (x_2-1)^2 + (x_4-1)^2 \right) \\
          & + 19.8(x_2-1)(x_4-1)

The idea is that by using AlgoPy to provide the gradient and hessian
of the objective function,
the nonlinear optimization procedures in
`scipy.optimize <http://docs.scipy.org/doc/scipy/reference/optimize.html>`_
will more easily find the values of :math:`x_1, x_2, x_3, x_4`
that minimize :math:`f(x_1, x_2, x_3, x_4)`.

For what it's worth, here is the symbolic Hessian according to Sympy::

    [1200*x1**2 - 400*x2 + 2, -400*x1,                       0,       0]
    [                -400*x1,   220.2,                       0,    19.8]
    [                      0,       0, 1080*x3**2 - 360*x4 + 2, -360*x3]
    [                      0,    19.8,                 -360*x3,   200.2]

And here is the python code for the minimization:

.. literalinclude:: wood_function_minimization.py


Here is its output::

    properties of the function at a local min:
    point:
    [ 1.  1.  1.  1.]
    function value:
    0.0
    autodiff gradient:
    [ 0.  0.  0.  0.]
    finite differences gradient:
    [ 0.  0.  0.  0.]
    autodiff hessian:
    [[  8.02000000e+02  -4.00000000e+02   0.00000000e+00   0.00000000e+00]
     [ -4.00000000e+02   2.20200000e+02   2.84217094e-14   1.98000000e+01]
     [  0.00000000e+00   2.84217094e-14   7.22000000e+02  -3.60000000e+02]
     [  0.00000000e+00   1.98000000e+01  -3.60000000e+02   2.00200000e+02]]
    finite differences hessian:
    [[  8.02000000e+02  -4.00000000e+02   0.00000000e+00   0.00000000e+00]
     [ -4.00000000e+02   2.20200000e+02  -1.62261147e-15   1.98000000e+01]
     [  0.00000000e+00  -1.62261147e-15   7.22000000e+02  -3.60000000e+02]
     [  0.00000000e+00   1.98000000e+01  -3.60000000e+02   2.00200000e+02]]

    ---------------------------------------------------------
    searches beginning from the easier init point [ 0.1  0.2  0.3  0.4]
    ---------------------------------------------------------

    properties of the function at the initial guess:
    point:
    [ 0.1  0.2  0.3  0.4]
    function value:
    33.163
    autodiff gradient:
    [ -9.4    9.96 -34.88  27.84]
    finite differences gradient:
    [ -9.4    9.96 -34.88  27.84]
    autodiff hessian:
    [[ -66.   -40.     0.     0. ]
     [ -40.   220.2    0.    19.8]
     [   0.     0.   -44.8 -108. ]
     [   0.    19.8 -108.   200.2]]
    finite differences hessian:
    [[ -6.60000000e+01  -4.00000000e+01   2.57962551e-16   2.41707021e-14]
     [ -4.00000000e+01   2.20200000e+02  -5.21231467e-18   1.98000000e+01]
     [  2.57962551e-16  -5.21231467e-18  -4.48000000e+01  -1.08000000e+02]
     [  2.41707021e-14   1.98000000e+01  -1.08000000e+02   2.00200000e+02]]

    strategy: default (Nelder-Mead)
    options: default
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 262
             Function evaluations: 438
    [ 0.99999926  0.9999881   1.00000257  1.00000442]

    strategy: ncg
    options: default
    gradient: autodiff
    hessian: autodiff
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 24
             Function evaluations: 36
             Gradient evaluations: 24
             Hessian evaluations: 24
    [ 1.  1.  1.  1.]

    strategy: ncg
    options: default
    gradient: autodiff
    hessian: finite differences
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 24
             Function evaluations: 36
             Gradient evaluations: 804
             Hessian evaluations: 0
    [ 1.          1.00000001  1.          1.        ]

    strategy: bfgs
    options: default
    gradient: autodiff
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 26
             Function evaluations: 32
             Gradient evaluations: 32
    [ 1.  1.  1.  1.]

    strategy: bfgs
    options: default
    gradient: finite differences
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 26
             Function evaluations: 192
             Gradient evaluations: 32
    [ 0.99999981  0.99999964  1.00000006  1.00000014]

    strategy: slsqp
    options: default
    gradient: autodiff
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: 1.35388134687e-08
                Iterations: 20
                Function evaluations: 33
                Gradient evaluations: 20
    [ 0.99999577  0.99999202  1.00001936  1.00004226]

    strategy: slsqp
    options: default
    gradient: finite differences
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: 1.34090220511e-08
                Iterations: 20
                Function evaluations: 133
                Gradient evaluations: 20
    [ 0.99999558  0.99999166  1.00001942  1.00004239]

    strategy: powell
    options: default
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 16
             Function evaluations: 798
    [ 1.  1.  1.  1.]

    strategy: tnc
    options: default
    gradient: autodiff
    (array([ 0.99999992,  0.99999983,  1.00000003,  1.00000008]), 65, 1)

    strategy: tnc
    options: default
    gradient: finite differences
    (array([ 1.00004027,  1.00008853,  0.99997404,  0.9999528 ]), 100, 3)


    ---------------------------------------------------------
    searches beginning from the more difficult init point [-3. -1. -3. -1.]
    ---------------------------------------------------------

    properties of the function at the initial guess:
    point:
    [-3. -1. -3. -1.]
    function value:
    19192.0
    autodiff gradient:
    [-12008.  -2080. -10808.  -1880.]
    finite differences gradient:
    [-12008.  -2080. -10808.  -1880.]
    autodiff hessian:
    [[  1.12020000e+04   1.20000000e+03   0.00000000e+00   0.00000000e+00]
     [  1.20000000e+03   2.20200000e+02   3.69482223e-13   1.98000000e+01]
     [  0.00000000e+00   3.69482223e-13   1.00820000e+04   1.08000000e+03]
     [  0.00000000e+00   1.98000000e+01   1.08000000e+03   2.00200000e+02]]
    finite differences hessian:
    [[  1.12020000e+04   1.20000000e+03   0.00000000e+00   1.97248577e-13]
     [  1.20000000e+03   2.20200000e+02   0.00000000e+00   1.98000000e+01]
     [  0.00000000e+00   0.00000000e+00   1.00820000e+04   1.08000000e+03]
     [  1.97248577e-13   1.98000000e+01   1.08000000e+03   2.00200000e+02]]

    strategy: default (Nelder-Mead)
    options: default
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 314
             Function evaluations: 527
    [ 0.99999777  0.99999832  1.00000621  1.0000122 ]

    strategy: ncg
    options: default
    gradient: autodiff
    hessian: autodiff
    Warning: Maximum number of iterations has been exceeded.
             Current function value: 2.246607
             Iterations: 800
             Function evaluations: 1462
             Gradient evaluations: 800
             Hessian evaluations: 800
    [ 1.35551493  1.83632796 -0.34501194  0.12565911]

    strategy: ncg
    options: default
    gradient: autodiff
    hessian: finite differences
    Warning: Maximum number of iterations has been exceeded.
             Current function value: 6.891168
             Iterations: 800
             Function evaluations: 1346
             Gradient evaluations: 6878
             Hessian evaluations: 0
    [ 0.10238992  0.01459057 -1.35571408  1.84742307]

    strategy: bfgs
    options: default
    gradient: autodiff
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 75
             Function evaluations: 89
             Gradient evaluations: 89
    [ 1.00000001  1.00000001  0.99999999  0.99999999]

    strategy: bfgs
    options: default
    gradient: finite differences
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 79
             Function evaluations: 564
             Gradient evaluations: 94
    [ 0.99999982  0.99999965  1.00000006  1.00000013]

    strategy: slsqp
    options: default
    gradient: autodiff
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: 7.87669164283
                Iterations: 14
                Function evaluations: 25
                Gradient evaluations: 14
    [-0.98976664  0.98975655 -0.94720029  0.90853426]

    strategy: slsqp
    options: default
    gradient: finite differences
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: 7.87669164153
                Iterations: 14
                Function evaluations: 95
                Gradient evaluations: 14
    [-0.98976671  0.98975668 -0.94720027  0.90853419]

    strategy: powell
    options: default
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 14
             Function evaluations: 656
    [ 1.  1.  1.  1.]

    strategy: tnc
    options: default
    gradient: autodiff
    (array([ 1.01054318,  1.02115755,  0.98918878,  0.97849078]), 100, 3)

    strategy: tnc
    options: default
    gradient: finite differences
    (array([ 1.33628967,  1.77114944,  0.10704316,  0.00221154]), 100, 3)



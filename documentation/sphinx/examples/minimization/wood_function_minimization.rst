Minimization of Wood's Function
-------------------------------

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
    searches beginning from the easier init point [ 1.1  1.2  1.3  1.4]
    ---------------------------------------------------------

    properties of the function at the initial guess:
    point:
    [ 1.1  1.2  1.3  1.4]
    function value:
    11.283
    autodiff gradient:
    [   4.6     9.96  136.32  -40.16]
    finite differences gradient:
    [   4.6     9.96  136.32  -40.16]
    autodiff hessian:
    [[  9.74000000e+02  -4.40000000e+02   0.00000000e+00   0.00000000e+00]
     [ -4.40000000e+02   2.20200000e+02   2.84217094e-14   1.98000000e+01]
     [  0.00000000e+00   2.84217094e-14   1.32320000e+03  -4.68000000e+02]
     [  0.00000000e+00   1.98000000e+01  -4.68000000e+02   2.00200000e+02]]
    finite differences hessian:
    [[  9.74000000e+02  -4.40000000e+02   1.00438116e-13   0.00000000e+00]
     [ -4.40000000e+02   2.20200000e+02   6.53681225e-14   1.98000000e+01]
     [  1.00438116e-13   6.53681225e-14   1.32320000e+03  -4.68000000e+02]
     [  0.00000000e+00   1.98000000e+01  -4.68000000e+02   2.00200000e+02]]

    strategy: default (Nelder-Mead)
    options: default
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 146
             Function evaluations: 249
    [ 0.99999164  0.99998319  1.00001086  1.00002861]

    strategy: ncg
    options: default
    gradient: autodiff
    hessian: autodiff
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 10
             Function evaluations: 11
             Gradient evaluations: 10
             Hessian evaluations: 10
    [ 1.00000012  1.00000024  1.          1.00000001]

    strategy: ncg
    options: default
    gradient: autodiff
    hessian: finite differences
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 10
             Function evaluations: 11
             Gradient evaluations: 54
             Hessian evaluations: 0
    [ 1.00000012  1.00000024  1.          1.00000001]

    strategy: cg
    options: default
    gradient: autodiff
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 33
             Function evaluations: 71
             Gradient evaluations: 71
    [ 1.00000139  1.00000278  0.99999861  0.99999721]

    strategy: cg
    options: default
    gradient: finite differences
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 58
             Function evaluations: 749
             Gradient evaluations: 123
    [ 0.99999733  0.99999467  1.0000027   1.00000542]

    strategy: bfgs
    options: default
    gradient: autodiff
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 15
             Function evaluations: 22
             Gradient evaluations: 22
    [ 0.99999999  0.99999997  1.00000001  1.00000002]

    strategy: bfgs
    options: default
    gradient: finite differences
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 15
             Function evaluations: 132
             Gradient evaluations: 22
    [ 0.9999998   0.99999962  1.00000007  1.00000016]

    strategy: slsqp
    options: default
    gradient: autodiff
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: 3.35065040984e-07
                Iterations: 12
                Function evaluations: 23
                Gradient evaluations: 12
    [ 1.0002004   1.0004415   0.99979783  0.99959924]

    strategy: slsqp
    options: default
    gradient: finite differences
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: 3.34830087144e-07
                Iterations: 12
                Function evaluations: 83
                Gradient evaluations: 12
    [ 1.00020021  1.00044114  0.9997979   0.99959938]

    strategy: powell
    options: default
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 16
             Function evaluations: 724
    [ 1.  1.  1.  1.]

    strategy: tnc
    options: default
    gradient: autodiff
    (array([ 0.99999805,  0.99999608,  1.00000183,  1.00000366]), 36, 1)

    strategy: tnc
    options: default
    gradient: finite differences
    (array([ 0.99999969,  0.99999926,  1.00000022,  1.0000005 ]), 72, 1)


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

    strategy: cg
    options: default
    gradient: autodiff
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 55
             Function evaluations: 110
             Gradient evaluations: 110
    [ 0.99999999  0.99999998  0.99999999  0.99999999]

    strategy: cg
    options: default
    gradient: finite differences
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 102
             Function evaluations: 1182
             Gradient evaluations: 197
    [ 1.00000232  1.00000467  0.99999761  0.99999522]

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



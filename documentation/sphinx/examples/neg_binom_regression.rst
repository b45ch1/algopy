Negative Binomial Regression
----------------------------

In this example we want to use AlgoPy to help compute the
maximum likelihood estimates and standard errors of parameters
of a nonlinear model.
This follows the
`statsmodels <http://statsmodels.sourceforge.net/>`_
generic maximum likelihood
`example
<http://statsmodels.sourceforge.net/
devel/examples/generated/example_gmle.html>`_
which uses the
`medpar
<http://vincentarelbundock.github.com/Rdatasets/doc/COUNT/medpar.html>`_
dataset.

.. math::

    \mathcal{L}(\beta_j; y, \alpha) = \sum_{i=1}^n y_i \log & 
    \left( \frac{\alpha \exp(X'_i \beta)}{1 + \alpha \exp(X'_i \beta)} \right)
    - \frac{1}{\alpha} \log \left( 1 + \alpha \exp(X'_i \beta) \right) \\
    & + \log \Gamma(y_i + 1/\alpha) -
        \log \Gamma(y_i + 1) - \log \Gamma(1/\alpha)

Here is the python code:

.. literalinclude:: neg_binom_regression.py


Here is its output::

    Optimization terminated successfully.
             Current function value: 4797.476603
             Iterations: 10
             Function evaluations: 11
             Gradient evaluations: 10
             Hessian evaluations: 10
    search results:
    [ 2.31027893  0.22124897  0.70615882 -0.06795522 -0.12906544  0.44575671]

    aic:
    9606.95320507

    standard error using observed fisher information,
    with hessian computed using algopy:
    [ 0.06794736  0.05059255  0.07613111  0.05326133  0.06854179  0.01981577]

    standard error using observed fisher information,
    with hessian computed using numdifftools:
    [ 0.06794736  0.05059255  0.07613111  0.05326133  0.06854179  0.01981577]


The agreement between this output and the statsmodels example results
suggests that the statsmodels
`caveat
<http://statsmodels.sourceforge.net/
devel/examples/generated/example_gmle.html#numerical-precision>`_
about numerical precision may not be the result of
numerical problems with the derivatives,
but rather that the R MASS
`implementation
<http://cran.r-project.org/web/packages/MASS/index.html>`_
may be giving less precise standard error estimates
or may not be using the observed fisher information to get the
standard error estimates in the most straightforward way.


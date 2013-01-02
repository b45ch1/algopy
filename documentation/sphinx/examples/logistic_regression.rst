Logistic Regression
----------------------------

In this example we want to use AlgoPy to help compute the
maximum likelihood estimates for a nonlinear model.
It is based on this
`question <http://scicomp.stackexchange.com/questions/4826>`_
on the scicomp stackexchange.


Here is the python code:

.. literalinclude:: logistic_regression.py


Here is its output::

    hardcoded good values:
    [-0.10296645 -0.0332327  -0.01209484  0.44626211  0.92554137  0.53973828
      1.7993371   0.7148045 ]

    neg log likelihood for good values:
    102.173732637


    hardcoded okay values:
    [-0.1  -0.03 -0.01  0.44  0.92  0.53  1.8   0.71]

    neg log likelihood for okay values:
    104.084160515


    maximum likelihood estimates:
    [-0.10296655 -0.0332327  -0.01209484  0.44626209  0.92554133  0.53973824
      1.79933696  0.71480445]

    neg log likelihood for maximum likelihood estimates:
    102.173732637


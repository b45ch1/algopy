Posterior Log Probability
-------------------------

We want the derivative of a posterior log probability density calculation.
We have a normal distribution with known variance.

.. literalinclude:: posterior_log_probability.py
   :lines: 1-

as output one obtains::
    
    walter@wronski$ python examples/posterior_log_probability.py
    function evaluation =
    134.752794884
    function evaluation + 1st directional derivative =
    [[ 134.75279488]
     [  49.85643448]]
    finite differences derivative =
    49.8564304507


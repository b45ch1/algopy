Linear Error Propagation
=========================


This example shows how ALGOPY can be used for linear error propagation.

Consider the error model

.. math::
    y = x + \epsilon

where :math:`x` a vector and :math:`\epsilon` a random vector that is normally distributed with zero mean and
covariance matrix :math:`\Sigma^2`. The :math:`y` is the observed quantity and :math:`x` is a real vector
representing the "true" value.

One defines some estimator :math:`\hat x` for :math:`x`, e.g. the arithmetic 
mean :math:`\hat x = \sum_{i=1}^{N_m} y_i`.
We assume that confidence region of the estimate :math:`\hat x` is known and has an
associated confidence region described by its covariance matrix 

.. math::
    \Sigma^2 = \mathbb E[(\hat x - E[\hat x])(\hat x - E[\hat x])^T]

The question is:  What can we say about the confidence region of the function
:math:`f(y)` when the confidence region of :math:`y` is described by the 
covariance matrix :math:`\Sigma^2`?

.. math::
    f: \mathbb R^N \rightarrow \mathbb R^M \\
       \hat x \mapsto \hat x = f(\hat x)


For affine (linear) functions

.. math::
    z = f(y) = Ay + b
        
the approach is described in the wikipedia article http://en.wikipedia.org/wiki/Propagation_of_uncertainty .
Nonlinear functions are simply linearized about the estimate :math:`\hat y` of :math:`\mathbb E[y]`.
In the vicinity of :math:`\hat y`, the linear model approximates the nonlinear function often quite well.
To linearize the function, the Jacobian :math:`J(\hat y)` of the function :math:`f(\hat y)` has to be computed, i.e.:

.. math::
    z \approx f(y) = f(\hat y) + J(\hat y) (y - \hat y)

The covariance matrix of :math:`z` is defined as 

.. math::
    C = \mathbb E[z z^T] = \mathbb E[ J y y^T J^T] = J \Sigma^2 J^T \; .
    
That means if we know :math:`J(y)`, we can approximately compute the confidence region if
:math:`f(\hat y)` is sufficiently linear.

To compute the Jacobian one can use the forward and the reverse mode of AD:
In the forward mode of AD one computes Nm directional derivatives, i.e. P = Nm.
In the reverse mode of AD one computes M adjoint derivatives, i.e. Q = M.


.. literalinclude:: error_propagation.py


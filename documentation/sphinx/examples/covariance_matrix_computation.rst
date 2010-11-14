Covariance Matrix Computation
=============================

In this example it is the goal to compute the gradient of one element 
of the covariance matrix :math:`C` of a constrained parameter estimation problem,
i.e,

.. math::
    \nabla_{J_1, J_2} y
    
where :math:`y = C_{11}`.
The covariance matrix satisfies the equation

.. math::
    C = \begin{pmatrix} I & 0 \end{pmatrix} \
        \begin{pmatrix} J_1^T J_1 & J_2^T \\ J_2 & 0 \end{pmatrix}^{-1} \
        \begin{pmatrix}  I \\ 0 \end{pmatrix}

where :math:`J_1` and :math:`J_2`.

Two possibilities are compared:
    1) filling a big matrix with elements, then invert it and return a view of
       of the upper left part of the matrix
    
    2) Computation of the Nullspace of J2 with a QR decomposition.
       The formula is :math:`C = Q_2^T( Q_2 J_1^T J1 Q_2^T)^{-1} Q_2`.

       Potentially, using the QR decomposition twice, i.e. once to compute :math:`Q_2` and
       then for :math:`J_1` compute :math:`Q_2^T` to avoid the multiplication which would square the condition
       number, may be numerically more stable. This has not been tested yet though.

The setup for both possibilities to compute the covariance matrix and their derivatives is the same

.. literalinclude:: covariance_matrix_computation.py
   :lines: 1-25

where:
    * D - 1 is the degree of the Taylor polynomial
    * P directional derivatives at once
    * M number of rows of J1
    * N number of cols of J1
    * K number of rows of J2 (must be smaller than N)
 
At first the naive function evaluation is traced.

.. literalinclude:: covariance_matrix_computation.py
   :lines: 27-36
   
then the nullspace method is traced
    
.. literalinclude:: covariance_matrix_computation.py
   :lines: 38-47
   
After that, the function evaluation is stored in the CGraph instance and can be used
to compute the gradient.

.. literalinclude:: covariance_matrix_computation.py
   :lines: 52-

   
One obtains the output::
    
    naive approach: dy/dJ1 =  [[ 1.09167163 -0.37815832 -0.45414733]
    [ 0.57524052 -0.19926504 -0.23930634]
    [-1.6063055   0.55642903  0.66824064]
    [-0.1674705   0.05801228  0.06966956]
    [-1.23363017  0.42733318  0.51320363]]
    naive approach: dy/dJ2 =  [[ 0.1039174  -0.0359973  -0.04323077]]
    nullspace approach: dy/dJ1 =  [[ 1.09167163 -0.37815832 -0.45414733]
    [ 0.57524052 -0.19926504 -0.23930634]
    [-1.6063055   0.55642903  0.66824064]
    [-0.1674705   0.05801228  0.06966956]
    [-1.23363017  0.42733318  0.51320363]]
    nullspace approach: dy/dJ2 =  [[ 0.1039174  -0.0359973  -0.04323077]]

As one can see, both methods yield the same result.

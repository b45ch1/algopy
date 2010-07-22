Covariance Matrix Computation
=============================


In this example it is the goal to compute derivatives of the covariance matrix
of a constrained parameter estimation problem.

I.e. compute

.. math::
    C = \begin{pmatrix} I & 0 \end{pmatrix} \
        \begin{pmatrix} J_1^T J_1 & J_2^T \\ J_2 & 0 \end{pmatrix}^{-1} \
        \begin{pmatrix}  I \\ 0 \end{pmatrix}

where :math:`J_1 = J_1(x)` and :math:`J_2 = J_2(x)` and  :math:`x \in \mathbb R^{N_x}`.

Two possibilities are compared:
    1) filling a big matrix with elements, then invert it and return a view of
       of the upper left part of the matrix
    
    2) Computation of the Nullspace of J2 with a QR decomposition.
       The formula is :math:`C = Q_2^T( Q_2 J_1^T J1 Q_2^T)^{-1} Q_2`.

       Potentially, using the QR decomposition twice, i.e. once to compute :math:`Q_2` and
       then for :math:`J_1` compute :math:`Q_2^T` to avoid the multiplication which would square the condition
       number, may be numerically more stable. This has not been tested yet though.

The corresponding code is

.. literalinclude:: covariance_matrix_computation.py








 

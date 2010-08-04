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

The setup for both possibilities to compute the covariance matrix and their derivatives is the same

.. literalinclude:: covariance_matrix_computation.py
   :lines: 1-4

where:
    * D - 1 is the degree of the Taylor polynomial
    * P directional derivatives at once
    * M number of rows of J1
    * N number of cols of J1
    * K number of rows of J2 (must be smaller than N)


Possibility 1: Nullspace approach
-----------------------------------

We now initialize an instance `cg` of the class CGraph and wrap the `UTPM` instances `J1` and `J2`
in `Function` instances to record the evaluation.
    
.. literalinclude:: covariance_matrix_computation.py
   :lines: 5-10
   
After that we perform the actual computation

.. literalinclude:: covariance_matrix_computation.py
   :lines: 13-20

   
One can see that this looks almost like a normal function evaluation with the only difference that we use the name `qr_full` to indicate that the
output matrix `Q` is quadratic and not rectangular as it returned by `numpy.linalg.qr`. When all computations are done, we call the `cg1.trace_off()`,
i.e. we stop recording computations. It is called nullspace method since `Q2` spans the nullspace of `J2`.

To be able to use the reverse mode of AD, we need to specify the independent and dependent variables.

.. literalinclude:: covariance_matrix_computation.py
   :lines: 22-23
   

Possibility 2: Imagespace approach
----------------------------------
One can compute the covariance matrix similarly in a direct way. We show the complete code:

.. literalinclude:: covariance_matrix_computation.py
   :lines: 29-42

Computing Derivatives in the Reverse Mode
-----------------------------------------
Now that the computations of the nullspace and imagespace method are stored in the computational graphs `cg1` and `cg2` we can compute derivatives
in the reverse mode of AD.

To be explicit, we want to compute

.. math::
    \frac{\partial C_{23}}{\partial J_1} \in \mathbb R^{M \times N} \quad \mbox{and} \quad \frac{\partial C_{23}}{\partial J_2} \in \mathbb R^{K \times N} \;,
    
i.e. the sensitivities of :math:`C_{23}` w.r.t. to all entries of :math:`J_1` and :math:`J_2`.    
That means, according to the theory of the reverse mode, we have to initialize :math:`\bar C` as follows:

.. literalinclude:: covariance_matrix_computation.py
   :lines: 47-48

All we have to do now is call the `CGraph.pullback` methods and extract the result

.. literalinclude:: covariance_matrix_computation.py
   :lines: 56-57

We get an output like::
    
    dC_23/dJ1=
    [[  0.24250447  -0.48870482  -0.07427832]
     [  1.91206665  -6.95700351   6.11249738]
     [  0.93224057  -2.9988799    2.13194142]
     [  6.01809283 -22.19748542  19.88783672]
     [ -2.02188653   8.22798775  -8.34415394]]
    dC_23/dJ2=
    [[ -3.67278871  12.211396    -9.25515817]]



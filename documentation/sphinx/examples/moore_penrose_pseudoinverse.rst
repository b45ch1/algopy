Computation of a Moore-Penrose Pseudoinverse
============================================

In this example it is the goal to compute derivatives of the Moore-Penrose pseudoinverse.
We compare different possibilities:

    1) A naive approach where A^T A is explicitly computed (numerically unstable)
    2) A QR approach where at first a QR decomposition of A is formed and the inverse
       is computed by a forward and then back substitution of R. T
    3) A direct approach where an analytic formula for the derivatives of the
       Moore-Penrose Formula is derived. Then usage of the QR decomposition is
       used to make the computation of the analytical formula numerically stable.

More explicitly, we compute

.. math::

    A^\dagger = (A^T A)^{-1} A^T

where :math:`A \in \mathbb R^{M \times N}` with :math:`M \geq N` with possibly
bad condition number but full column rank.

.. literalinclude:: moore_penrose_pseudoinverse.py


Polarization Identities for Mixed Partial Derivatives
-----------------------------------------------------

This is an advanced tutorial. It explains how functions containing derivatives of
other functions can be evaluated using univariate Taylor polynomial arithmetic 
by use of polarization identities.
The described technique could be implemented as convenience functions just as
UTPM.init_jacobian, UTPM.init_hessian, etc. 

Consider the function 

.. math::
    F: \Bbb{R}^N \to & \Bbb{R}^M \\
    x  \mapsto& y = F(x) = \begin{pmatrix} x_1 x_2 \\ x_2 x_3 \\ x_1 - x_2 \end{pmatrix}

where :math:`(N,N)=(3,3)` and we want to compute

.. math::
    \nabla_x \Phi(J(x)) =& \mathrm{tr} ( J(x)^T J(x)) \;,
    
where :math:`J(x) = {\partial F \over \partial x}(x)`.
In univariate Taylor polynomial arithmetic it is possible to compute the :math:`i`-th
column of :math:`J(x)` using :math:`{\partial \over \partial T} F(x + e_i)|_{T=0}`.
Also, it is possible to compute the :math:`j`-th element of the gradient
:math:`\nabla_x \Phi(J(x))` using
:math:`{\partial \over \partial T} \Phi( J(x + e_j)) |_{T=0}`.
That means, in total it is necessary to compute

.. math::
   {\partial \Phi \over \partial x_j} (J(x)) =& \left. {\partial \over \partial T_2} \Phi \left(
   \begin{pmatrix}
   {\partial \over \partial T_1} F(x + e_1 T_1 + e_jT_2), 
   \dots, 
   {\partial \over \partial T_1} F(x + e_N T_1 + e_jT_2) \\
   \end{pmatrix}
   \right) \right|_{T_1 = T_2 =0} \\
   =& \left. \sum_{m=1}^M \sum_{n=1}^N {\partial \Phi \over \partial J_{mn}} \right|_{J = J(x)} 
   {\partial^2 \over \partial T_2 \partial T_1} F_m(x + e_n T_1 + e_jT_2)

In this form it seems to be necessary to propagate a multivariate Taylor polynomial
:math:`x + e_1 T_1 + e_jT_2` through :math:`F`. However, one can use a
polarization identity like

.. math::
    \left. {\partial^2 \over \partial T_1 \partial T_2} F(x + e_i T_1 + e_jT_2)
    \right|_{T_1 = T_2 = 0}
    = \left. {1 \over 4} {\partial^2 \over \partial T^2} \left( F(x + (e_i + e_j) T) - F(x + (e_i - e_j)T) \right) \right|_{T=0} 

to cast the problem back to a problem where twice the number of univariate Taylor
polynomials have to be propagated.
In total we need to cycle over :math:`e_i`, for :math:`i =1,2,\dots,N`
and :math:`e_j`, for :math:`j = 1,2,\dots,N`, which can both be written as columns/rows
of the identity matrix.
Stacking :math:`e_i + e_j` and :math:`e_i - e_j` for all possible choices one obtains
the univariate directions::

    dirs =
    [[ 2.  0.  0.]
     [ 0.  0.  0.]
     [ 1.  1.  0.]
     [ 1. -1.  0.]
     [ 1.  0.  1.]
     [ 1.  0. -1.]
     [ 1.  1.  0.]
     [-1.  1.  0.]
     [ 0.  2.  0.]
     [ 0.  0.  0.]
     [ 0.  1.  1.]
     [ 0.  1. -1.]
     [ 1.  0.  1.]
     [-1.  0.  1.]
     [ 0.  1.  1.]
     [ 0. -1.  1.]
     [ 0.  0.  2.]
     [ 0.  0.  0.]]

As one can see, some directions are zero and could be avoided.
Without this obvious improvement, there are :math:`P = 2N^2=18` directions.
Now to how the above example can be computed in AlgoPy.

.. literalinclude:: polarization.py
   :lines: 0-

As output we obtain::
    
    $ python polarization.py 
    dirs =
    [[ 2.  0.  0.]
     [ 0.  0.  0.]
     [ 1.  1.  0.]
     [ 1. -1.  0.]
     [ 1.  0.  1.]
     [ 1.  0. -1.]
     [ 1.  1.  0.]
     [-1.  1.  0.]
     [ 0.  2.  0.]
     [ 0.  0.  0.]
     [ 0.  1.  1.]
     [ 0.  1. -1.]
     [ 1.  0.  1.]
     [-1.  0.  1.]
     [ 0.  1.  1.]
     [ 0. -1.  1.]
     [ 0.  0.  2.]
     [ 0.  0.  0.]]
    Phi= [[ 20.  20.  20.]
     [  2.   8.   6.]]
    gradient of Phi = [ 2.  8.  6.]

which is the correct value.




One should note that actually in this special case, when all elements
:math:`{\partial \Phi \over \partial x_j}` have to be propagated
it is advantageous to use another polarization identity which requires to 
propagate only as many directions as the Hessian has distinct elements.
However, the polarization identity from above is more flexible since it allows to
compute mixed partial derivatives more directly.




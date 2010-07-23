Differentiation of ODE Solutions
================================

It is very easy to use AD techniques to obtain derivatives of the form :math:`\frac{d x(t)}{d p}`,
where :math:`x(t) \equiv x(t; x_0, p) \in \mathbb R^{N_x}` is solution of the ordinary differential equation

.. math::
    \dot x(t) = f(t, x, p) \\
      x(0) = x_0(p) \;,
      
where the initial values :math:`x(0)` is a function :math:`x_0(p)` depending on
some parameter :math:`p \in \mathbb R^{N_p}`.


Consider the following code that computes  :math:`\frac{d x(t)}{d p}` of the 
harmonic oscillator described by the ODE

.. math::
    \dot x(t) = \begin{pmatrix} x_2 \\ -p  x_1 \end{pmatrix} \;.

To illustrate the idea, we use the
explict Euler integration scheme. More sophisticated methods are similarly easy.
The idea is simply to perform all computation in UTP arithmetic.

.. literalinclude:: explicit_euler.py

The generated plot shows the numerically computed trajectory
and the analytically derived solutions.  One can see that the numerical trajectory
of :math:`\frac{d x(t)}{d p}` is close to the analytical solution. More elaborate
ODE integrators would yield better results.


.. image:: explicit_euler.png
    :align: center
    :scale: 100



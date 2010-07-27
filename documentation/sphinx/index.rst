.. algopy documentation master file, created by
   sphinx-quickstart on Sun Jul 18 16:23:52 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ALGOPY, Algorithmic Differentiation in Python
=============================================




Documentation:

.. toctree::
   :maxdepth: 2
   
   getting_started.rst
   datastructure_and_algorithms.rst
   examples_tracer.rst
   
Documented Examples:

.. toctree::
   :maxdepth: 2
   
   examples/covariance_matrix_computation.rst
   examples/error_propagation.rst
   examples/moore_penrose_pseudoinverse.rst
   examples/ode_solvers.rst
   examples/comparison_forward_reverse_mode.rst

What is ALGOPY?
---------------

The purpose of ALGOPY is the evaluation of higher-order derivatives
in the forward and reverse mode of Algorithmic Differentiation (AD). Particular
focus are functions that contain numerical linear algebra functions as
they often appear in statistically motivated functions.



How does it work?:
------------------

The central idea of ALGOPY is the computation on (univariate) Taylor polynomials with with matrix coefficients.
these are implemented as (class) methods of `algopy.UTPM`.

To allow the use of the reverse mode of AD a simple code tracer has been implemented in
`algopy.tracer`. The idea is to record the computation procedure in a datastructure s.t.
the control flow sequence can walked in reverse order.

Current Issues:
---------------

    * broadcasting for UTPM instances is not implemented yet. i.e.
      x = UTPM(numpy.random.rand((2,3,4))
      y = dot(x,x) + x
      will most likely raise an error
      
    * some algorithms require vectors to be columns of a matrix.
      I.e. if x is a vector it should be initialized as
      x = UTPM(numpy.random.rand(D,P,N,1) and not as
      UTPM(numpy.random.rand(D,P,N)) as one would typically do it using numpy.
      
    * there is no vectorized reverse mode yet. That means that one can compute 
      columns of a Jacobian of dimension (M,N) by propagating N directional 
      derivatives at once. In the reverse mode one would like to propagate M
      adjoint directions at once. However, this is not implemented yet, i.e. one
      has to repeat the procedure M times.
      



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


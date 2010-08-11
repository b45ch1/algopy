.. algopy documentation master file, created by
   sphinx-quickstart on Sun Jul 18 16:23:52 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ALGOPY, Algorithmic Differentiation in Python
=============================================




Documentation:

.. toctree::
   :maxdepth: 1
   
   datastructure_and_algorithms.rst
   examples_tracer.rst
   
Simple Examples:

.. toctree::
   :maxdepth: 1
   
   examples/series_expansion.rst
   examples/first_order_forward.rst
   
Advanced Examples:

.. toctree::
   :maxdepth: 1
   
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

If the input UTPs are correctly initialized one can interpret the coefficients of
the resulting polynomial as higher-order derivatives.

To allow the use of the reverse mode of AD a simple code tracer has been implemented in
`algopy.tracer`. The idea is to record the computation procedure in a datastructure s.t.
the control flow sequence can walked in reverse order.

There has been a winterschool for Algorithmic Differentiation where some tutorials
are a good introduction to what ALGOPY does.

http://www.sintef.no/Projectweb/eVITA/English/eSCience-Meeting-2010/Winter-School/


Getting Started:
----------------









Example 3: First-order directional Derivatives through Numerical Linear Algebra Functions
-----------------------------------------------------------------------------------------

ALGOPY can be used to compute series expansions through complicated functions that also contain numerical linear algebra functions.
Consider the contrived example that appears in similar form in statistically
motivated functions. It is the goal to compute derivatives of the function `y = f(x)`::

    import numpy; from numpy import log, exp, sin, cos
    import algopy; from algopy import UTPM, dot, inv, zeros
    
    def f(x):
        A = zeros((2,2),dtype=x)
        A[0,0] = numpy.log(x[0]*x[1])
        A[0,1] = numpy.log(x[1]) + exp(x[0])
        A[1,0] = sin(x[1])**2 + cos(x[0])**3.1
        A[1,1] = x[0]**cos(x[1])
        return log( dot(x.T,  dot( inv(A), x)))
    
    
    x = UTPM(zeros((2,1,2),dtype=float))
    x.data[0,0] = [1,2]
    x.data[1,0] = [1,0]
    y = f(x)
    
    print 'normal function evaluation f(x) = ',y.data[0,0]
    print 'directional derivative df/dx1 = ',y.data[1,0]


Current Issues:
---------------
      
    * some algorithms require vectors to be columns of a matrix.
      I.e. if x is a vector it should be initialized as
      x = UTPM(numpy.random.rand(D,P,N,1) and not as
      UTPM(numpy.random.rand(D,P,N)) as one would typically do it using numpy.
      
    * there is no vectorized reverse mode yet. That means that one can compute 
      columns of a Jacobian of dimension (M,N) by propagating N directional 
      derivatives at once. In the reverse mode one would like to propagate M
      adjoint directions at once. However, this is not implemented yet, i.e. one
      has to repeat the procedure M times.
      
Unit Test
---------

ALGOPY uses the same testing facilitities as NumPy. I.e., one can run the complete
unit test with::
    
    $ python -c "import algopy; algopy.test()"


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


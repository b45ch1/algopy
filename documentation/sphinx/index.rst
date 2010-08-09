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



Example 1: univariate Taylor Series Expansions
-----------------------------------------------

As an easy example we want to compute the Taylor series expansion of

.. math::
    y = f(x) = \sin(\cos(x) + \sin(x))
    
about :math:`x_0 = 0.3`. The first thing to notice is that we can as well compute the
Taylor series expansion of

.. math::
    y = f(x_0 + t) = \sin(\cos(x_0 + t) + \sin(x_0 + t))
    
about :math:`t = 0`. Taylor's theorem yields

.. math::
    f(x_0 + t) &= \sum_{d=0}^{D-1} y_d t^d + R_{D}(t) \\
    \mbox{where } \quad y_d &= \left. \frac{1}{d!} \frac{d^d }{d t^d}f(x_0 + t) \right|_{t = 0} \;.
    
and :math:`R_D(x)` is the remainder term.

Slightly rewritten one has

.. math::
    y(t) = f(x(t)) + \mathcal O(t^D)
    
i.e., one has a polynomial :math:`x(t) = \sum_{d=0}^{D-1} x_d t^d` as input and
computes a polynomial :math:`y(t) = \sum_{d=0}^{D-1} y_d t^d + \mathcal O(t^d)` as output.

This is now formulated in a way that can be used with ALGOPY.
    
.. literalinclude:: index.py
    :lines: 0-13

Don't be confused by the P. It can be used to evaluate several Taylor series expansions
at once. The important point to notice is that the D in the code is the same D
as in the formula above. I.e., it is the number of coefficients in the polynomials.
The important point is

.. warning:: The coefficients of the univariate Taylor polynomial (UTP) are stored in
          the attribute UTPM.data. It is a x.ndarray with shape (D,P) + shape of the coefficient.
          In this example, the coefficients :math:`x_d` are scalars and thus x.data.shape = (D,P).
          However, if the the coefficients were vectors of size N, then x.data.shape would be (D,P,N), 
          and if the coefficients were matrices with shape (M,N), then x.data.shape would be (D,P,M,N).
          
          


To see that ALGOPY indeed computes the correct Taylor series expansion we plot
the original function and the Taylor polynomials evaluated at different orders.

.. literalinclude:: index.py
    :lines: 14-34


.. figure:: taylor_approximation.png
    :align: center
    :scale: 50
    
    This plot shows Taylor approximations of different orders. 
    The point :math:`x_0 = 0.3` is plotted as a red dot and the original
    function is plotted as black dots. One can see that the higher the order,
    the better the approximation.
    


Example 2: Forward Mode of Algorithmic Differentiation
------------------------------------------------------
In this example we want to show how one can extract derivatives
from the computed univariate Taylor polynomials (UTP). For simplicity we only show
first-order derivatives but ALGOPY also supports the computation of higher-order
derivatives by an interpolation-evaluation approach.

The basic observation is that by use of the chain rule one obtains functions
:math:`F: \mathbb R^N \rightarrow \mathbb R^M`

.. math::
    \left. \frac{d}{d t} F(x_0 + x_1 t) \right|_{t=0} = \left. \frac{d}{d x} f(x) \right|_{x = x_0} \cdot x_1\;.

i.e. a Jacobian-vector product.

Again, we look a simple contrived example and we want to compute the first column
of the Jacobian, i.e., :math:`x_1 = (1,0,0)`.

.. literalinclude:: index.py
    :lines: 35-60

As output one gets::
    
    y0 =  [  3.  15.   5.]
    y  =  [[[  3.  15.   5.]]
    
     [[  3.   0.   5.]]]
    y.shape = (3,)
    y.data.shape = (2, 1, 3)
    dF/dx(x0) * x1 = [ 3.  0.  5.]


and the question is how to interpret this result. First off, y0 is just the usual
function evaluation using numpy but y represent a univariate Taylor polynomial (UTP).
One can see that each coefficient of the polynomial has the shape (3,). We extract
the directional derivative as the first coefficient of the UTP.

One can see that this is indeed the numerical value of first column of the Jacobian J(1,3,5)::
    
    def J(x):
        ret = numpy.zeros((3,3),dtype=float)
        ret[0,:] = [x[1], x[0],  0  ]
        ret[1,:] = [0,  , x[2], x[1]]
        ret[2,:] = [x[2],   0 , x[0]]



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


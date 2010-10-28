.. algopy documentation master file, created by
   sphinx-quickstart on Sun Jul 18 16:23:52 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ALGOPY, Algorithmic Differentiation in Python
=============================================

What is ALGOPY?
---------------

The purpose of ALGOPY is the evaluation of higher-order derivatives
in the forward and reverse mode of Algorithmic Differentiation (AD) of functions that are implemented as Python programs.
Particular focus are functions that contain numerical linear algebra functions as
they often appear in statistically motivated functions.

Current version is 0.2.3.

Help improve ALGOPY
-------------------
If you have any questions or suggestions please use the mailing list 
http://groups.google.com/group/algopy?msg=new&lnk=gcis
or alternatively write me an email(sebastian.walter@gmail.com). This will make it much easier for me to provide code/documentation that is easy to understand. Of course, you are also welcome to contribute code and bugfixes. For instance, an nice addition
would be a set of high-level functions that make it easier for new users to compute the gradient, Jacobian, Jacobian-vector, vector-Jacobian, Hessian, Hessian-vector.

Getting Started:
----------------
For the impatient, we show a minimalistic example how ALGOPY can be used to compute
a gradient in the forward mode of AD of a simple function and compare the result to the symbolically computed
gradient.

.. literalinclude:: getting_started.py
    :lines: 1-

Some words on what's going on: Instead of computing the function `f(x)` with a `x = numpy.array([3.,5.,7.])` as input we want to compute with a Univariate Taylor Polynomial (*UTP*) of degree 1. Operator overloading is used to redefine the functions `+,-,*,/,sin,cos,...` for `algopy.UTPM` instances. That means, when `x` is an `algopy.UTPM` instance then `f(x)` is also an `algopy.UTPM` instance.
The coefficients of the polynomial are stored in the attribute `data`. 

The integers `D=2,P=3,N=3` have the following meaning: The UTP has degree 1, i.e. `D=2` coefficients are necessary to describe it. `N=3` is the dimension of `x`. `P=3` allows us to evaluate three different Taylor polynomials at once, i.e., it vectorizes the operation.

If one executes that code one obtains as output::
    
    $ python getting_started.py 
    gradient computed with ALGOPY using UTP arithmetic =  [ 135.42768462   41.08553692   15.        ]
    evaluated symbolic gradient =  [ 135.42768462   41.08553692   15.        ]
    difference = [ 0.  0.  0.]

The derivative computed with ALGOPY is up to machine precision
the same as the symbolically computed gradient.
    
    
Example 2: First-order directional Derivatives through Numerical Linear Algebra Functions
-----------------------------------------------------------------------------------------

ALGOPY can not only be used to compute series expansions of simple functions
as shown above. A particular strenght of ALGOPY is that it allows to compute series
expansions through numerical linear algebra functions.
Consider the contrived example that appears in similar form in statistically
motivated functions. It is the goal to compute the directional derivative

.. math::
    \nabla_x f((3,5)) \cdot \begin{pmatrix} 7 \\ 11 \end{pmatrix}
    
The code is::

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
    x.data[0,0] = [3,5]
    x.data[1,0] = [7,11]
    y = f(x)
    
    print 'normal function evaluation f(x) = ',y.data[0,0]
    print 'directional derivative df/dx1 = ',y.data[1,0]


How does it work?:
------------------
ALGOPY offers the forward mode and the reverse mode of AD.

Forward Mode of AD:

    The basic idea is the computation on (univariate) Taylor polynomials
    with with matrix coefficients. More precisely, ALGOPY supports univariate Taylor
    polynomial (UTP) arithmetic where the coefficients of the polynomial are numpy.ndarrays.
    To distinguish Taylor polynomials from real vectors resp. matrices they are written with enclosing brackets:
    
    .. math::
        [x]_D = [x_0, \dots, x_{D-1}] = \sum_{d=0}^{D-1} x_d T^d \;,
        
    where each :math:`x_0, x_1, \dots` are arrays, e.g. a (5,7) array.
    This mathematical object is described by numpy.ndarray with shape (D,P, 5,7).
    The :math:`T` is an indeterminate, i.e. a formal/dummy variable. Roughly speaking, this is the UTP equivalent to the imaginary number :math:`i` in complex arithmetic. The `P` can be used to compute several Taylor expansions at once. I.e., a vectorization to avoid the recomputation of the same functions with different inputs.

    If the input UTPs are correctly initialized one can interpret the coefficients of
    the resulting polynomial as higher-order derivatives. Have a look at the `Taylor series expansion example`_
    for a more detailed discussion.


.. _`Taylor series expansion example`: ./examples/series_expansion.html


Reverse Mode of AD:

    To allow the use of the reverse mode of AD a simple code tracer has been implemented in
    `algopy.tracer`. The idea is to record the computation procedure in a datastructure s.t.
    the control flow sequence can walked in reverse order.
    There is no complete documentation for the reverse mode yet.

Further Reading
---------------

There has been a winterschool for Algorithmic Differentiation. Some tutorials explain Taylor polynomial arithmetic.
http://www.sintef.no/Projectweb/eVITA/English/eSCience-Meeting-2010/Winter-School/

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
   
Application Examples:

.. toctree::
   :maxdepth: 1
   
   examples/posterior_log_probability.rst
   examples/leastsquaresfitting.rst
   examples/hessian_of_potential_function.rst
   

Additional Information:

.. toctree::
   :maxdepth: 1
   
   datastructure_and_algorithms.rst
   examples_tracer.rst

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

Version Changelog
-----------------
* Version 0.2.2
    * fixed some broadcasting bugs with UTPM instances
    * fixed a bug in algopy.zeros

* Version 0.2.3
    * added UTPM.init_jacobian and UTPM.extract_jacobian
    * added UTPM.init_hessian and UTPM.extract_hessian
    * added UTPM.init_tensor and UTPM.extract_tensor
    * added UTPM.__len__, i.e. len(x) works now for x a UTPM instance
    * fixed a bug in algopy.dot(x,y) in the case when x is a numpy.ndarray and y is a UTPM instance

      
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


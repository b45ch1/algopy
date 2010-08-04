.. algopy documentation master file, created by
   sphinx-quickstart on Sun Jul 18 16:23:52 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ALGOPY, Algorithmic Differentiation in Python
=============================================




Documentation:

.. toctree::
   :maxdepth: 2
   
   datastructure_and_algorithms.rst
   examples_tracer.rst
   
Advanced Examples:

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



Example 1: univariate Taylor Series Expansions
-----------------------------------------------
ALGOPY can be used to compute Taylor series expansions of arbitrary complicated
functions. To show some easy examples from calculus, we look at the series expansion
of sin(t) and cos(t). The mathematical formula is given as:

.. math::
    \sin(t) &= x - \frac{t^3}{3!} + \frac{t^5}{5!} - \frac{t^7}{7!} + \cdots   = \sum_{d=0}^\infty \frac{(-1)^d}{(2d+1)!}t^{2d+1}\\
    \cos(t) &= 1 - \frac{t^2}{2!} + \frac{t^4}{4!} - \frac{t^6}{6!} + \cdots  = \sum_{d=0}^\infty \frac{(-1)^d}{(2d)!}t^{2d}

With ALGOPY you can compute series expansions of the form

.. math::
    y(t) &= \sin(x(t)) \;.
    
That means, to compute the series expansion of sine and cosine, one initializes
as follows.

.. math::
    x(t) = 0 + 1t + 0 t^2 + 0 t^3 + \dots + 0 t^{D-1} \equiv [x]_D
    

This mathematical problem is formulated in Python as follows::
    
    import numpy; from numpy import sin,cos
    from algopy import UTPM
    
    D = 5; P = 1
    x = UTPM(numpy.zeros((D,P)))
    x.data[1] = 1
    
    y1 = sin(x)
    y2 = cos(x)
    
    print 'series coefficients of y1 = ',y1.data[:,0]
    print 'series coefficients of y2 = ',y2.data[:,0]
    
Execution of the code yields the output::
    
    series coefficients of y1 =  [ 0.          1.          0.         -0.16666667  0.        ]
    series coefficients of y2 =  [ 1.         -0.         -0.5        -0.          0.04166667]

One can check that the computed coefficients are correct.

Newcomers to ALGOPY often find the quantity P confusing. It allows to compute
the same function with several inputs at once. For now we use P=1.

Computing the series expansion of 

.. math::
    y(t) = \sin(\cos(x(t)) + \sin(x(t)))
    
is similarily easy::
    
    z = sin(cos(x) + sin(x))
    print 'series coefficients of  z=',z.data[:,0]
    
produces::
    
    series coefficients of  z= [ 0.84147098  0.54030231 -0.69088665  0.24063472  0.22771075]

as output.




Example 2: First-order directional Derivatives
-----------------------------------------------

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


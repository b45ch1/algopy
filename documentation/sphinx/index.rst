.. algopy documentation master file, created by
   sphinx-quickstart on Sun Jul 18 16:23:52 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ALGOPY, Algorithmic Differentiation in Python
=============================================

What is ALGOPY?
---------------

The purpose of ALGOPY is the evaluation of higher-order derivatives
in the forward and reverse mode of Algorithmic Differentiation (AD). Particular
focus are functions that contain numerical linear algebra functions as
they often appear in statistically motivated functions.

Disclaimer
----------

At the moment, ALGOPY offers only rather low-level functions. I.e., the user needs
some background knowledge of AD and how ALGOPY works. This documentation should
explain most of what is necessary to know. Unfortunately, it is not very well structured at the moment.
So, if you have any questions or
suggestions don't hesitate to write me an email (sebastian.walter@gmail.com).
This will make it much easier for me to provide a text that is easy to understand.
Also, is planned to add a user-friendly API on top of the low-level functions to compute
the Jacobian, the Hessian and the gradient.


Getting Started:
----------------
For the impatient, we show a minimalistic example how ALGOPY can be used to compute
a gradient of a simple function in the forward mode of AD and compare the result to the symbolically computed
gradient.

.. literalinclude:: getting_started.py
    :lines: 1-

If one executes that code one obtains as output::
    
    $ python getting_started.py 
    gradient computed with ALGOPY using UTP arithmetic =  [ 135.42768462   41.08553692   15.        ]
    evaluated symbolic gradient =  [ 135.42768462   41.08553692   15.        ]
    difference = [ 0.  0.  0.]

We skip here an explanation of what exactly ALGOPY is doing internally here,
and just note that the derivative computed with ALGOPY is up to machine precision
the same as the symbolically computed gradient.

    
First-order directional Derivatives through Numerical Linear Algebra Functions
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

      
Unit Test
---------

ALGOPY uses the same testing facilitities as NumPy. I.e., one can run the complete
unit test with::
    
    $ python -c "import algopy; algopy.test()"



Further Documentation:
----------------------

The Forward Mode of Algorithmic Differentiation:

.. toctree::
   :maxdepth: 1
   
   examples/series_expansion.rst
   examples/first_order_forward.rst

The Reverse Mode of Algorithmic Differentiation:

    Todo
   
Advanced Examples:

.. toctree::
   :maxdepth: 1
   
   examples/covariance_matrix_computation.rst
   examples/error_propagation.rst
   examples/moore_penrose_pseudoinverse.rst
   examples/ode_solvers.rst
   examples/comparison_forward_reverse_mode.rst
   
   
Further Reading:

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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


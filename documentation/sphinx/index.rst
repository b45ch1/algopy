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


Help improve ALGOPY
-------------------
If you have any questions, suggestions or bug reports  please use the mailing list 
http://groups.google.com/group/algopy?msg=new&lnk=gcis
or alternatively write me an email(sebastian.walter@gmail.com). This will make it much easier for me to provide code/documentation that is easy to understand. 
Of course, you are also welcome to contribute code, bugfixes, examples, success stories ;), ...


Installation and Upgrade:
-------------------------

Current version is 0.3.0

Official releases:
    * available at:  http://pypi.python.org/pypi/algopy
    * if you have easy_install you can use the shell command
        - `$ easy_install algopy` for installation
        - `$ easy_install --upgrade algopy` to upgrade to the newest version

Bleeding edge:
    * the most recent version is available at https://github.com/b45ch1/algopy .
    * includes additional documentation, e.g. talks and additional examples and the sphinx *.rst documents
    
Dependencies:
    * numpy
    * scipy
    * (optional/recommended) nose
    * (optional/recommended) yapgvb (to generate plots of the computational graph)


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

Information on the web:

There has been a winterschool for Algorithmic Differentiation. Some tutorials explain Taylor polynomial arithmetic.
http://www.sintef.no/Projectweb/eVITA/English/eSCience-Meeting-2010/Winter-School/

Talks:
    * :download:`Informal talk at the IWR Heidelberg, April 29th, 2010<./talks/informal_talk_iwr_heidelberg_theory_and_tools_for_algorithmic_differentiation.pdf>`.
    * :download:`Univariate Taylor polynomial arithmetic applied to matrix factorizations in the forward and reverse mode of algorithmic differentiation, June 3rd, 2010, EuroAD in Paderborn<./talks/walter_euroad2010_paderborn_univariate_taylor_polynomial_arithmetic_applied_to_matrix_factorizations_in_the_forward_and_reverse_mode_of_algorithmic_differentiation.pdf>`.

    
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
   examples/minimal_surface.rst
   

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

* Version 0.3.0:
    * renamed push_forward to pushforward, this is more consistent w.r.t. the pullback
    * UTPM.__repr__ now returns a string of the form `UTPM(...)`
    * refactored the tracer: it should now be possible to trace the function evaluation with normal numpy.ndarrays. After that, one can use cg.pushforward with UTPM instances or call cg.gradient, etc.
    * UTPM.reshape is now a method, not a class method
    * added broadcasting support for __setitem__, iadd and isub
    * added Function.ndim
    * added preliminary complex numbers support for arithmetic with UTPM instances (reverse mode using the tracer is not supported yet)
    * UTPM.reshape now can also take integers as input, not only tuples of integers
    * added UTPM.tan, UTPM.arcsin, UTPM.arccos, UTPM.arctan, UTPM.sinh, UTPM.cosh, UTPM.tanh
    * made init_hessian and extract_hessian generic (to make it useful for complex valued functions)
    * added comparison operators <,>,<=,>=,== to UTPM
    * added UTPM.init_jac_vec and UTPM.extract_jac_vec
    
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


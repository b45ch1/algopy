.. algopy documentation master file, created by
   sphinx-quickstart on Sun Jul 18 16:23:52 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AlgoPy, Algorithmic Differentiation in Python
=============================================

What is AlgoPy?
---------------

The purpose of AlgoPy is the evaluation of higher-order derivatives
in the forward and reverse mode of Algorithmic Differentiation (AD) of functions
that are implemented as Python programs.
Particular focus are functions that contain numerical linear algebra functions as
they often appear in statistically motivated functions.
The intended use of AlgoPy is for easy prototyping at reasonable execution speeds.
More precisely, for a typical program a directional derivative takes order 10
times as much time as time as the function evaluation. This is approximately also
true for the gradient.


What can AlgoPy do for you?
----------------------------
    * evaluation of derivatives useful for nonlinear continuous optimization
        * gradient
        * Jacobian
        * Hessian
        * Jacobian vector product
        * vector Jacobian product
        * Hessian vector product
        * vector Hessian vector product
        * higher-order tensors

    * Taylor series evaluation
        * for modeling higher-order processes
        * could be used to compute Taylor series expansions for ODE/DAE integration.



Getting Started:
----------------
For the impatient, we show a minimalistic example how AlgoPy can be used to compute
derivatives.

.. literalinclude:: getting_started.py
    :lines: 1-

If one executes that code one obtains as output::

    $ python getting_started.py
    jacobian =  [ 135.42768462   41.08553692   15.        ]
    gradient = [array([ 135.42768462,   41.08553692,   15.        ])]
    Jacobian = [array([[ 135.42768462,   41.08553692,   15.        ]])]
    Hessian = [[[ 100.42768462   27.08553692    5.        ]
      [  27.08553692    0.            3.        ]
      [   5.            3.            0.        ]]]
    Hessian vector product = [ 567.13842308  126.34214769   35.        ]

Help improve AlgoPy
-------------------
If you have any questions, suggestions or bug reports  please use the mailing list
http://groups.google.com/group/algopy?msg=new&lnk=gcis
or alternatively write me an email(sebastian.walter@gmail.com).
This will make it much easier for me to provide code/documentation that
is easier to understand. Of course, you are also welcome to contribute code,
bugfixes, examples, success stories ;), ...


Issues and common pitfalls
---------------------------

.. warning::

    Do not use `numpy.ndarray` with `algopy.UTPM` or `algopy.Function` as
    elements!

    Example 1::

        import numpy, algopy

        x = algopy.Function(1)
        y = algopy.Function(2)

        # wrong
        z = numpy.array([x,y])

        # right
        z = algopy.zeros(2, dtype=x)
        z[0] = x
        z[1] = y

    Example 2::

        x = algopy.UTPM(numpy.ones((2,1)))
        y = algopy.UTPM(2*numpy.ones((2,1)))

        # wrong
        z = numpy.array([x,y])

        # right
        z = algopy.zeros(2, dtype=x)
        z[0] = x
        z[1] = y


.. warning::

    Broadcasting does not work in the reverse mode with NumPy 1.6!
    (it used to work for NumPy 1.4 ...)

    This is due to unexpected behavior of numpy, see discussion on
    http://thread.gmane.org/gmane.comp.python.numeric.general/51379


    Example::

        import numpy, algopy

        def f(x):
            N, = x.shape
            A = algopy.zeros((N,N), dtype=x)
            A[0,0] = x[2]
            A[1,2] = x[1]
            return algopy.sum(A*x)


        cg = algopy.CGraph()

        x = algopy.Function(numpy.ones(3))
        y = f(x)

        cg.independentFunctionList = [x]
        cg.dependentFunctionList = [y]

        print cg.gradient(numpy.array([1.,2.,3.]))

    Such code may give wrong results on your machine if you happen to have the
    wrong NumPy version.



Potential Improvements
----------------------
    * better memory management to speed things up
    * direct support for nested derivatives (requires the algorithms to be generic)
    * better complex number support, in particular also the reverse mode
    * support for sparse Jacobian and sparse Hessian computations using graph
      coloring as explained in http://portal.acm.org/citation.cfm?id=1096222


Related Work
------------

AlgoPy has been influenced by the following publications:

    * "ADOL-C: A Package for the Automatic Differentiation of Algorithms Written
      in C/C++", Andreas Griewank, David Juedes, H. Mitev, Jean Utke, Olaf Vogel,
      Andrea Walther
    * "Evaluating Higher Derivative Tensors by Forward Propagation of Univariate
      Taylor Series", Andreas Griewank, Jean Utke and Andrea Walther
    * "Taylor series integration of differential-algebraic equations:
      automatic differentiation as a tool for
      simulating rigid body mechanical systems", Eric Phipps, phd thesis
    * "Collected Matrix Derivative Results for Forward and Reverse Mode
      Algorithmic Differentiation", Mike Giles,
      http://www.springerlink.com/content/h1750t57160w2782/


Publications related to AlgoPy
------------------------------

* Walter et al.,
  "On evaluating higher-order derivatives of the QR decomposition of tall
  matrices with full column rank in forward and reverse mode algorithmic
  differentiation", `http://dx.doi.org/10.1080/10556788.2011.610454 <http://dx.doi.org/10.1080/10556788.2011.610454>`_

* Walter et al.,
  "Algorithmic differentiation in Python with AlgoPy", `http://dx.doi.org/10.1016/j.jocs.2011.10.007 <http://dx.doi.org/10.1016/j.jocs.2011.10.007>`_



Please cite the following paper if you use AlgoPy::

    @article{Walter2011,
    title = "Algorithmic differentiation in Python with AlgoPy",
    journal = "Journal of Computational Science",
    volume = "",
    number = "0",
    pages = " - ",
    year = "2011",
    note = "",
    issn = "1877-7503",
    doi = "10.1016/j.jocs.2011.10.007",
    url = "http://www.sciencedirect.com/science/article/pii/S1877750311001013",
    author = "Sebastian F. Walter and Lutz Lehmann",
    keywords = "Automatic differentiation",
    keywords = "Cholesky decomposition",
    keywords = "Hierarchical approach",
    keywords = "Higher-order derivatives",
    keywords = "Numerical linear algebra",
    keywords = "NumPy",
    keywords = "Taylor arithmetic"
    }




Installation and Upgrade:
-------------------------

Current version is 0.3.2

Official releases:
    * available at:  http://pypi.python.org/pypi/algopy
    * with pip
        - `$ pip install algopy`
    * with easy_install
        - `$ easy_install algopy` for installation
        - `$ easy_install --upgrade algopy` to upgrade to the newest version
    * NetBSD package
        - http://pkgsrc.se/wip/py-algopy
        - http://cvsweb.se.netbsd.org/cgi-bin/bsdweb.cgi/wip/py-algopy

Bleeding edge:
    * the most recent version is available at https://github.com/b45ch1/algopy
    * includes additional documentation, e.g. talks and additional examples and
      the sphinx `*.rst` documents


Dependencies:

    ALGOPY Core:
        * numpy
        * scipy

    ALGOPY Examples:
        * pyadolc

    Run tests:
        * Nose

    Documentation:
        * sphinx
        * matplotlib, mayavi2, yapgvb




How does it work?:
------------------
AlgoPy offers the forward mode and the reverse mode of AD.

Forward Mode of AD:

    The basic idea is the computation on (univariate) Taylor polynomials
    with with matrix coefficients. More precisely, AlgoPy supports univariate Taylor
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

There has been a winterschool for Algorithmic Differentiation.
Some tutorials explain Taylor polynomial arithmetic.
http://www.sintef.no/Projectweb/eVITA/Winter-Schools/2010/

Talks:
    * :download:`Informal talk at the IWR Heidelberg, April 29th, 2010<./talks/informal_talk_iwr_heidelberg_theory_and_tools_for_algorithmic_differentiation.pdf>`.
    * :download:`Univariate Taylor polynomial arithmetic applied to matrix factorizations in the forward and reverse mode of algorithmic differentiation, June 3rd, 2010, EuroAD in Paderborn<./talks/walter_euroad2010_paderborn_univariate_taylor_polynomial_arithmetic_applied_to_matrix_factorizations_in_the_forward_and_reverse_mode_of_algorithmic_differentiation.pdf>`.


Simple Examples:

.. toctree::
   :maxdepth: 1

   getting_started.rst
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
   examples/ampl_minimization_problem.rst
   examples/rosenbrock_banana.rst
   examples/himmelblau_minimization.rst

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
   examples/polarization.rst
   symbolic_differentiation.rst


How is AlgoPy organized:

.. toctree::
    :maxdepth: 1

    folder_structure.rst

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
    * added CGraph.function, CGraph.gradient, CGraph.hessian, CGraph.hess_vec

* Version 0.3.1
    * replaced algopy.sum by a faster implementation
    * fixed a bug in getitem of the UTPM instance: now works also with numpy.int64
      as index
    * added dedicated algopy.sum and algopy.prod
    * added UTPM.pb_sqrt
    * fixed bug in tracing operations involving neg(x)
    * added algopy.outer
    * changed API of CGraph.hessian, CGraph.jac_vec etc. One has now to write
      CGraph.jacobian(x) instead of CGraph.jacobian([x]).

* Version 0.3.2
    * improved error reporting in the reverse mode: when "something goes wrong"
      in cg.gradient([1.,2.,3.]) one now gets a much more detailed traceback
    * added A.reshape(...) support to the reverse mode
    * improved support for broadcasting for UTPM instances

Unit Test
---------

AlgoPy uses the same testing facilitities as NumPy. I.e., one can run the complete
unit test with::

    $ python -c "import algopy; algopy.test()"


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


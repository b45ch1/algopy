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
   examples_tracer.rst
   
Documented Examples:

.. toctree::
   :maxdepth: 2
   
   examples/covariance_matrix_computation.rst
   examples/error_propagation.rst


What is ALGOPY?
---------------

The purpose of ALGOPY is the evaluation of higher-order derivatives
in the forward and reverse mode of Algorithmic Differentiation (AD). Particular
focus are functions that contain numerical linear algebra functions as
they often appear in statistically motivated functions.








Rationale:
----------

The central idea of ALGOPY is the computation on Taylor polynomials with scalar
coefficients and with matrix coefficients. These algorithms are primarily used for
Algorithmic Differentiation (AD)in the forward and reverse mode.

The focus are univariate Taylor polynomials over matrices (UTPM),implemented in
the class `algopy.utpm.UTPM`.

To allow the use of the reverse mode of AD a simple code tracer has been implemented in
`algopy.tracer`. The idea is to record the computation procedure in a datastructure s.t.
the control flow sequence can walked in reverse order.

ALGOPY is a research prototype where, to the best of authors'
knowledge, some algorithms are implemented that cannot be found elsewhere.

Most of ALGOPY is implemented in pure Python. However, some submodules are implemented
in pure C. For these submodules Python bindings using ctypes are provided.

The algorithms are quite well-tested and have been successfully used.
However, since the user-base is currently quite small, it is possible that bugs
may still be persistent.
Also, not all submodules of ALGOPY are feature complete. The most important parts
`algopy.tracer` and `algopy.upt.UTPM` are however fairly complete.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


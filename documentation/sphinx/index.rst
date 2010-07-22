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



How does it work?:
------------------

The central idea of ALGOPY is the computation on (univariate) Taylor polynomials with with matrix coefficients.
At the moment, also other possibilities exist, but the class `algopy.UTPM` is the main focus of ALGOPY.

To allow the use of the reverse mode of AD a simple code tracer has been implemented in
`algopy.tracer`. The idea is to record the computation procedure in a datastructure s.t.
the control flow sequence can walked in reverse order.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


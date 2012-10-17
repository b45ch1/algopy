=================================
AlgoPy file and folder structure
=================================
---------------------------------
an overview
---------------------------------


The general idea
================

This document explains how AlgoPy is structured.

* The submodule tracer contains two classes

  - CGraph
  - Function

  CGraph stands for computational graph and is a directed acyclic graph.
  A node in the CGraph is an instance of Function.

  It is called "tracer" since the sequence of operations to evaluate some
  function::

      def eval_f(x):
          ...
          return y

  is recorded and stored in a CGraph instance.

  Using CGraph.gradient, CGraph.jacobian, etc. one can evaluate derivatives
  in the forward and reverse mode of AD.

  The function `CGraph.pullback` cycles in reverse direction through the
  sequence of operations and calls at each Node the method
  `Function.pullback`, which in turn calls one of the classmethods
  `UTPM.pb_*`.


* The submodule UTPM contains an implementation of the algebraic class
  of matrix Taylor polynomials.
  It is an "extension" of numpy.ndarray.


algopy/tests/test_*.py
======================

Normally, for each file `file.py` there exists a
`tests/test_file.py`.

In this folder there are tests that check the functionality
as seen by a user.

* test_globalfuncs.py checks that algopy.dot, algopy.sin etc.
  correctly call UTPM.sin, Function.sin etc.

* test_linalg.py checks that forward and reverse mode yields
  the same results for linear algebra functions

* test_special.py checks that forward and reverse mode yields
  the same results for linear algebra functions

* test_operators.py checks that forward and reverse mode yields
  the same results for operators __add__, __mul__, __iadd__, etc.
  Also check whether broadcasting works correctly.

* test_examples.py contains more complex examples







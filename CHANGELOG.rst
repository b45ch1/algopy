**Version 0.7.1**

* Support for numpy==2.0.0. Dropping support for numpy 1.x.x.

**Version 0.6.0**

* Updated Algopy to support recent versions of numpy==1.26.4 and scipy==1.13.1
* Scipy has removed support for confluent hypergeometric functions. Thus, the support of these functions in Algopy has been also removed in this version.
* Numpy has removed support for nosetests. Thus, algopy now uses pytest to run tests via `pytest -v algopy`

**Version 0.5.0**

* add Python 3 compatibility
* add Travis CI

**Version 0.4.0**

* added support for a variety of new functions, mostly contributed by
  Alex Griffing, NCSU:
  expm, hyp1f1, hyperu, hyp2f0, polygamma, psi, erf, erfi, dawsn, logit, expit

**Version 0.3.2**

* improved error reporting in the reverse mode: when "something goes wrong"
  in cg.gradient([1.,2.,3.]) one now gets a much more detailed traceback
* added A.reshape(...) support to the reverse mode
* improved support for broadcasting for UTPM instances

**Version 0.3.1**

* replaced algopy.sum by a faster implementation
* fixed a bug in getitem of the UTPM instance: now works also with numpy.int64
  as index
* added dedicated algopy.sum and algopy.prod
* added UTPM.pb_sqrt
* fixed bug in tracing operations involving neg(x)
* added algopy.outer
* changed API of CGraph.hessian, CGraph.jac_vec etc. One has now to write
  CGraph.jacobian(x) instead of CGraph.jacobian([x]).

**Version 0.3.0**

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

**Version 0.2.3**

* added UTPM.init_jacobian and UTPM.extract_jacobian
* added UTPM.init_hessian and UTPM.extract_hessian
* added UTPM.init_tensor and UTPM.extract_tensor
* added UTPM.__len__, i.e. len(x) works now for x a UTPM instance
* fixed a bug in algopy.dot(x,y) in the case when x is a numpy.ndarray and y is a UTPM instance

**Version 0.2.2**

* fixed some broadcasting bugs with UTPM instances
* fixed a bug in algopy.zeros











The Code Tracer
================

ALGOPY features a simple code tracer. A trace of the executed code is necessary
for the reverse mode of AD. The basic idea is to wrap an object `x` in an
`algopy.Function` instance. Performing operations with `Function` instances
are recorded in a computational graph, implemented in `algopy.CGraph`.

For example::

    >>> import numpy
    >>> from algopy import CGraph, Function
    >>> cg = CGraph()
    >>> cg.trace_on()
    >>> x = Function(1)
    >>> y = Function(3)
    >>> z = x * y + x
    >>> cg.trace_off()
    >>> cg.independentFunctionList = [x,y]
    >>> cg.dependentFunctionList = [z]
    >>> print cg
    >>> cg.plot('example_tracer_cgraph.png')
    
    
    
    Id: IDs: 0 <- [0]
    x:
        1 
    
    
    Id: IDs: 1 <- [1]
    x:
        3 
    
    
    __mul__: IDs: 2 <- [0, 1]
    x:
        3 
    
    
    __add__: IDs: 3 <- [2, 0]
    x:
        4 
    
    Independent Function List:
    [0, 1]
    
    Dependent Function List:
    [3]


and the plotted computational graph is:

.. image:: example_tracer_cgraph.png
    :align: center
    :scale: 100



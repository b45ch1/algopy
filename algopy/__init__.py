"""
ALGOPY, a library for Automatic Differentation (AD) in Python
-------------------------------------------------------------

Rationale:
    ALGOPY is a research prototype striving to provide state of the art algorithms.
    It is not (yet) geared towards end users.
    The ultimative goal is to provide high performance algorithms
    that can be used to differentiate dynamic systems  (ODEs, DAEs, PDEs)
    and static systems (linear/nonlinear systems of equations).
    
    ALGOPY focuses on the algebraic differentiation of elementary operations,
    e.g. C = dot(A,B) where A,B,C are matrices, y = sin(x), z = x*y, etc.
    to compute derivatives of functions composed of such elementary functions.
    
    In particular, ALGOPY offers:
        
        Univariate Taylor Propagation:
            
            * Univariate Taylor Propagation on Scalars  (UTPS)
              Implementation in: `./algopy/utp/utps.py`
            * Univariate Taylor Propagation on Matrices (UTPM)
              Implemenation in: `./algopy/utp/utpm.py`
            * Exact Interpolation of Higher Order Derivative Tensors:
              (Hessians, etc.)
              
        Reverse Mode:
        
            ALGOPY also features functionality for convenient differentiation of a given
            algorithm. For that, the sequence of operation is recorded by tracing the 
            evaluation of the algorithm. Implementation in: `./algopy/tracer.py`

    ALGOPY aims to provide algorithms in a clean and accessible way allowing quick
    understanding of the underlying algorithms. Therefore, it should be easy to
    port to other programming languages, take code snippets.
    If optimized algorithms are wanted, they should be provided in a subclass derived
    from the reference implementation.


"""

import utp
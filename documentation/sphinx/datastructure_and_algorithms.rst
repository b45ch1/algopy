Datastructure and Algorithms

The mathematical object is a Univariate Taylor Polynomial over Matrices (UTPM)

.. math::
    [x]_D = [x_0, \dots, x_{D-1}] = \sum_{d=0}^{D-1} x_d T^d \;,
    
where each :math:`x_d` is some array, e.g. a (5,7) array.
This mathematical object is described by numpy.ndarray with shape (D,P, 5,7).
P>1 triggers a vectorized function evaluation.

All algorithms are implemented
in the following fashion::
    
    def add(x_data, y_data, z_data):
        z_data[...] = x_data[...] + y_data[...]
        
where the inputs `x_data,y_data` are  numpy.ndarray's and `add` changes the elements
of the numpy.ndarray z. I.e., the algorithms are implemented in a similar way as LAPACK or Fortran functions
in general. One can find the UTPM algorithms in `algopy/utpm/algorithms.py` where they are class functions
of a mixin class.

In practice, working with such algorithms is cumbersome. ALGOPY therefore also
offers the class `algopy.UTPM` which is a thin wrapper around the algorithms 
and provides overloaded functions and operators. The data is saved in the attribute UTPM.data.

The following code shows how to use the algorithms
directly and using the syntactic sugar provided by the UTPM class.

.. literalinclude:: datastructure_and_algorithms.py



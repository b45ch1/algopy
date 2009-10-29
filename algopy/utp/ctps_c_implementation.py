import os
import ctypes
import numpy
from numpy.ctypeslib import ndpointer

from algopy.base_type import GradedRing

_ctps = numpy.ctypeslib.load_library('libctps', os.path.dirname(__file__))

dvec = ndpointer(dtype=numpy.float64, ndim=1, flags='CONTIGUOUS,ALIGNED')

_ctps.add.argtypes = [ctypes.c_int, dvec, dvec, dvec ]
_ctps.crossmultwise.argtypes = [ctypes.c_int, dvec, dvec, dvec ]


class CTPS_C(GradedRing):
    def __init__(self, data):
        """
        CTPS = Cross Derivative Taylor Polynomial
        Implements the factor ring  R[t1,...,tK]/<t1^2,...,tK^2>
        
        Calls C functions internally.
        """
        self.data = numpy.array(data)
        
    @classmethod
    def zeros_like(cls, data):
        return numpy.zeros_like(data)

    @classmethod
    def mul(cls, retval_data, lhs_data, rhs_data):
        K = 100000
        _ctps.crossmultwise(K, lhs_data, rhs_data, retval_data)


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.data) 

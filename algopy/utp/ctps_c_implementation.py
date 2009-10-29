import os
import ctypes
import numpy
from numpy.ctypeslib import ndpointer

from algopy.base_type import GradedRing

_ctps = numpy.ctypeslib.load_library('libctps', os.path.dirname(__file__))

double_ptr =  ctypes.POINTER(ctypes.c_double)
argtypes = [ctypes.c_int, double_ptr, double_ptr, double_ptr]

_ctps.add.argtypes = argtypes
_ctps.crossmultwise.argtypes = argtypes

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
    def add(cls, retval_data, lhs_data, rhs_data):
        K = retval_data.size
        _ctps.add(K,
        lhs_data.ctypes.data_as(double_ptr),
        rhs_data.ctypes.data_as(double_ptr),
        retval_data.ctypes.data_as(double_ptr))

    @classmethod
    def mul(cls, retval_data, lhs_data, rhs_data):
        K = retval_data.size
        _ctps.crossmultwise(K,
        lhs_data.ctypes.data_as(double_ptr),
        rhs_data.ctypes.data_as(double_ptr),
        retval_data.ctypes.data_as(double_ptr))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.data) 

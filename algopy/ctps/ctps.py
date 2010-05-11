import os
import ctypes
import numpy

from algopy.base_type import Ring

_ctps = numpy.ctypeslib.load_library('libctps', os.path.dirname(__file__))

double_ptr =  ctypes.POINTER(ctypes.c_double)
argtypes1 = [ctypes.c_int, double_ptr, double_ptr, double_ptr]

_ctps.ctps_add.argtypes = argtypes1
_ctps.ctps_sub.argtypes = argtypes1
_ctps.ctps_mul.argtypes = argtypes1
_ctps.ctps_div.argtypes = argtypes1

class CTPS(Ring):
    def __init__(self, data):
        """
        CTPS = Cross Derivative Taylor Polynomial
        Implements the factor ring  R[t1,...,tK]/<t1^2,...,tK^2>
        
        Calls C functions internally. I.e. functionality *should* be the same as for the class CTPS.
        """
        self.data = numpy.array(data)
        
    @classmethod
    def __scalar_to_data__(cls, xdata, x):
        xdata[0] = x
        
    @classmethod
    def __zeros_like__(cls, data):
        return numpy.zeros_like(data)

    @classmethod
    def add(cls, retval_data, lhs_data, rhs_data):
        K = retval_data.size
        _ctps.ctps_add(K,
        lhs_data.ctypes.data_as(double_ptr),
        rhs_data.ctypes.data_as(double_ptr),
        retval_data.ctypes.data_as(double_ptr))
        
    @classmethod
    def sub(cls, retval_data, lhs_data, rhs_data):
        K = retval_data.size
        _ctps.ctps_sub(K,
        lhs_data.ctypes.data_as(double_ptr),
        rhs_data.ctypes.data_as(double_ptr),
        retval_data.ctypes.data_as(double_ptr))        

    @classmethod
    def mul(cls, retval_data, lhs_data, rhs_data):
        K = retval_data.size
        _ctps.ctps_mul(K,
        lhs_data.ctypes.data_as(double_ptr),
        rhs_data.ctypes.data_as(double_ptr),
        retval_data.ctypes.data_as(double_ptr))
        
    @classmethod
    def div(cls, retval_data, lhs_data, rhs_data):
        K = retval_data.size
        _ctps.ctps_div(K,
        lhs_data.ctypes.data_as(double_ptr),
        rhs_data.ctypes.data_as(double_ptr),
        retval_data.ctypes.data_as(double_ptr))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.data) 

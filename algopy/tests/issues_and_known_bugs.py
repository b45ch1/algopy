from numpy.testing import *
import os

from algopy.tracer.tracer import *
from algopy.utp.utpm import UTPM
from algopy.utp.utps import UTPS

import numpy

class Tracer_Issues(TestCase):
    def test_simple_repeated_buffered_operation(self):
        """
        test:  y *= x
        
        by y = ones(...)
           y[...] *= x
           
        """
        D,P = 1,1
        
        cg = CGraph()
        x = UTPM(numpy.random.rand(*(D,P)))
        Fx = Function(x)
        Fy = Function(UTPM(numpy.ones((D,P))))
        Fy[...] *= Fx
        cg.independentFunctionList = [Fx]
        cg.dependentFunctionList = [Fy]
        
        assert_array_almost_equal(Fy.x.data[0], x.data[0])
        cg.push_forward([x])
        assert_array_almost_equal(Fy.x.data[0], x.data[0])



if __name__ == "__main__":
    run_module_suite()

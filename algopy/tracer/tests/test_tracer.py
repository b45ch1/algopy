from numpy.testing import *
from algopy.tracer.tracer import *

import numpy

class Test_Function(TestCase):
    def test_init(self):
        x = 1.
        fx = Function(x)
        
    def test_push_forward(self):
        fx = Function(1.)
        fy = Function(2.)
        
        Function.push_forward(numpy.add, (fx,fy))
        
    def test_push_forward2(self):
        fx = Function(numpy.random.rand(3,3))
        Function.push_forward(numpy.linalg.qr, fx)

class Test_CG(TestCase):
    def test_push_forward(self):
        cg = CG()
        fx = Function(1.)
        fy = Function(2.)
        
        Function.push_forward(numpy.add, (fx,fy))
        
        print cg

if __name__ == "__main__":
    run_module_suite()

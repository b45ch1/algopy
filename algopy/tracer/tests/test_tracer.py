from numpy.testing import *
from algopy.tracer.tracer import *

import numpy

class Test_Function_on_numpy_types(TestCase):
    
    def test_init(self):
        x = 1.
        fx = Function(x)
        assert_array_almost_equal(fx.x,x)
        
    def test_push_forward_add(self):
        x,y = 1.,2.
        fx = Function(x)
        fy = Function(y)
        
        fz = Function.push_forward(numpy.add, (fx,fy))
        assert_almost_equal(fz.x, x + y)
        
    def test_push_forward_qr(self):
        x = numpy.random.rand(3,3)
        fx = Function(x)
        fy = Function.push_forward(numpy.linalg.qr, fx)
        y  = numpy.linalg.qr(x)
        assert_array_almost_equal(fy.x, y)
        
class Test_Function_on_UTPM(TestCase):
    
    def test_init(self):
        D,P = 3,4
        x = UTPM(numpy.ones((D,P)))
        
        

class Test_CG(TestCase):
    def test_push_forward(self):
        cg = CG()
        fx = Function(1.)
        fy = Function(2.)
        
        Function.push_forward(numpy.add, (fx,fy))
        
        print cg

if __name__ == "__main__":
    run_module_suite()

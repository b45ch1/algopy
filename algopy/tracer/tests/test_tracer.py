from numpy.testing import *
from algopy.tracer.tracer import *
from algopy.utp.utpm import UTPM

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
        
    def test_push_forward_add(self):
        D,P,N,M = 2,3,4,5
        x = UTPM(numpy.random.rand(D,P,N,M))
        y = UTPM(numpy.random.rand(D,P,N,M))
        fx = Function(x)
        fy = Function(y)
        
        fz = Function.push_forward(UTPM.add, (fx,fy))
        assert_almost_equal(fz.x.data, (x + y).data)        
        
        

class Test_CG(TestCase):
    def test_push_forward(self):
        cg = CG()
        fx = Function(1.)
        fy = Function(2.)
        
        fz = Function.push_forward(numpy.add, (fx,fy))
        fz = Function.push_forward(numpy.multiply, (fz,fy))

        cg.independentFunctionList = [fx,fy]
        cg.dependentFunctionList = [fz]
        
        x = 32.23
        y = 235.
        cg.push_forward([x,y])
        assert_array_almost_equal( cg.dependentFunctionList[0].x,  (x + y) * y)
        
        
if __name__ == "__main__":
    run_module_suite()

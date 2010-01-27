from numpy.testing import *
from algopy.tracer.tracer import *
from algopy.utp.utpm import UTPM
from algopy.utp.utps import UTPS

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
        
        
    def test_pullback_add(self):
        D,P,N,M = 2,3,4,5
        x = UTPM(numpy.random.rand(D,P,N,M))
        y = UTPM(numpy.random.rand(D,P,N,M))
        fx = Function(x)
        fy = Function(y)
        
        fz = Function.push_forward(UTPM.add, (fx,fy))
        fz.xbar = fz.x.zeros_like()
        fx.xbar = fx.x.zeros_like()
        fy.xbar = fy.x.zeros_like()
        
        fz = Function.pullback(fz)
        assert_almost_equal(fx.xbar.data, (fz.xbar * fy.xbar).data)
        assert_almost_equal(fy.xbar.data, (fz.xbar * fx.xbar).data)
        

class Test_CGgraph_on_numpy_operations(TestCase):
    def test_push_forward(self):
        cg = CGraph()
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
        
        
class TestCGraph_on_UTPS(TestCase):
    def test_forward(self):
        cg = CGraph()
        ax = UTPS([3.,1.])
        ay = UTPS([7.,0.])
        fx = Function(ax)
        fy = Function(ay)
        fv1 = fx * fy
        fv2 = (fv1 * fx + fy)*fv1
        cg.independentFunctionList = [fx,fy]
        cg.dependentFunctionList = [fv2]
        cg.push_forward([ax,ay])
        assert_array_almost_equal(cg.dependentFunctionList[0].x.data, ((ax*ay * ax + ay)*ax*ay).data)        
        
        
class Test_CGgraph_on_UTPM(TestCase):
    def test_push_forward(self):
        cg = CGraph()
        D,P,N,M = 2,5,7,11
        aX = UTPM(numpy.random.rand(D,P,N,M))
        aY = UTPM(numpy.random.rand(D,P,N,M))
        fX = Function(aX)
        fY = Function(aY)
        fV1 = fX * fY
        fV2 = (fV1 * fX + fY)*fV1
        cg.independentFunctionList = [fX,fY]
        cg.dependentFunctionList = [fV2]
        cg.push_forward([aX,aY])
        assert_array_almost_equal(cg.dependentFunctionList[0].x.data, ((aX*aY * aX + aY)*aX*aY).data)
        
    def test_pullback(self):
        cg = CGraph()
        D,P,N,M = 1,1,1,1
        aX = UTPM(numpy.random.rand(D,P,N,M))
        aY = UTPM(numpy.random.rand(D,P,N,M))
        fX = Function(aX)
        fY = Function(aY)
        fV1 = fX * fY
        fV2 = (fV1 * fX + fY)*fV1
        cg.independentFunctionList = [fX,fY]
        cg.dependentFunctionList = [fV2]
        cg.push_forward([aX,aY])
        
        aV2bar = fV2.x.zeros_like()
        cg.pullback([aV2bar])
        
        print cg


if __name__ == "__main__":
    run_module_suite()

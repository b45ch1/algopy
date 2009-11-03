from numpy.testing import *
import numpy
import numpy.random

import algopy.tracer
import algopy.utp.utps
import algopy.utp.utpm

from algopy.tracer import *
from algopy.utp.utps import *
from algopy.utp.utpm import *
from algopy.utp.ctps_c import *


class TestFunction(TestCase):
    def test_function_constructor(self):

        class foo:
            pass

        cg = CGraph()
        fx = Function(2.)
        fy = Function(foo())
        fz = Function(fx)

    def test_add(self):
        cg = CGraph()
        fx = Function(2.)
        fy = Function(3.)
        fz = fx + fy
        assert_almost_equal(fz.x, fx.x + fy.x)

    def test_sub(self):
        cg = CGraph()
        fx = Function(2.)
        fy = Function(3.)
        fz = fx - fy
        assert_almost_equal(fz.x, fx.x - fy.x)

    def test_mul(self):
        cg = CGraph()
        fx = Function(2.)
        fy = Function(3.)
        fz = fx * fy
        assert_almost_equal(fz.x, fx.x * fy.x)

    def test_div(self):
        cg = CGraph()
        fx = Function(2.)
        fy = Function(3.)
        fz = fx / fy
        assert_almost_equal(fz.x, fx.x / fy.x)

class TestCGraphOnSclars(TestCase):
    def test_plotting_the_graph(self):
        import os.path
        cg = CGraph()
        fx = Function(3.)
        fy = Function(7.)
        fv1 = fx * fy
        fv2 = (fv1 * fx + fy)*fv1
        cg.independentFunctionList = [fx,fy]
        cg.dependentFunctionList = [fv2]
        cg.plot(filename = '/tmp/cgraph.png', method = None, orientation = 'TD')
        assert os.path.isfile('/tmp/cgraph.png')
        #os.remove('/tmp/cgraph.png')

    def test_forward(self):
        cg = CGraph()
        fx = Function(3.)
        fy = Function(7.)
        fv1 = fx * fy
        fv2 = (fv1 * fx + fy)*fv1
        cg.independentFunctionList = [fx,fy]
        cg.dependentFunctionList = [fv2]

        x = 23.
        y = 23523.
        cg.forward([x,y])
        assert_almost_equal(cg.dependentFunctionList[0].x, (x*y * x + y)*x*y)

    def test_reverse(self):
        cg = CGraph()
        x = 3.
        y = 7.
        fx = Function(x)
        fy = Function(y)
        fv1 = fx * fy
        fv2 = (fv1 * fx + fy)*fv1
        cg.independentFunctionList = [fx,fy]
        cg.dependentFunctionList = [fv2]

        v2bar = 1.
        cg.reverse([v2bar])

        xbar_reverse = cg.independentFunctionList[0].xbar
        ybar_reverse = cg.independentFunctionList[1].xbar

        xbar_symbolic = 3 * x**2 * y**2 + y**2
        ybar_symbolic = 2*x**3 * y + 2 * x * y

        assert_almost_equal(xbar_reverse, xbar_symbolic)
        assert_almost_equal(ybar_reverse, ybar_symbolic)


class TestCGraphOnUTPS(TestCase):
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
        cg.forward([ax,ay])
        assert_array_almost_equal(cg.dependentFunctionList[0].x.tc, ((ax*ay * ax + ay)*ax*ay).tc)

    def test_reverse(self):
        cg = CGraph()
        ax = UTPS([3.,1.])
        ay = UTPS([7.,0.])
        fx = Function(ax)
        fy = Function(ay)
        fv1 = fx * fy
        fv2 = (fv1 * fx + fy)*fv1
        cg.independentFunctionList = [fx,fy]
        cg.dependentFunctionList = [fv2]

        v2bar = UTPS([1.,0.])
        cg.reverse([v2bar])

        xbar_reverse = cg.independentFunctionList[0].xbar
        ybar_reverse = cg.independentFunctionList[1].xbar
        
        xbar_symbolic = 3 * ax**2 * ay**2 + ay**2
        ybar_symbolic = 2*ax**3 * ay + 2 * ax * ay

        # print xbar_symbolic.tc
        # print xbar_reverse
        # print ybar_symbolic
        # print ybar_reverse
        
        assert_array_almost_equal(xbar_reverse.tc, xbar_symbolic.tc)
        assert_array_almost_equal(ybar_reverse.tc, ybar_symbolic.tc)


class TestCGraphOnUTPM(TestCase):

    def test_plotting_the_graph(self):
        import os.path
        cg = CGraph()
        X = 2 * numpy.random.rand(2,2,2,2)
        Y = 2 * numpy.random.rand(2,2,2,2)
        AX = UTPM(X)
        AY = UTPM(Y)
        FX = Function(AX)
        FY = Function(AY)

        FX = FX*FY
        FX = FX.dot(FY) + FX.transpose()
        FX = FY + FX * FY
        FY = FX.inv()
        FY = FY.transpose()
        FZ = FX * FY
        FW = Function([[FX, FZ], [FZ, FY]])
        FTR = FW.trace()
        cg.independentFunctionList = [FX, FY]
        cg.dependentFunctionList = [FTR]
        cg.plot(filename = '/tmp/cgraph_matrix.png', method = None, orientation = 'TD')
        assert os.path.isfile('/tmp/cgraph_matrix.png')
        #os.remove('/tmp/cgraph.png')

    def test_forward(self):
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
        cg.forward([aX,aY])
        assert_array_almost_equal(cg.dependentFunctionList[0].x.tc, ((aX*aY * aX + aY)*aX*aY).tc)

    def test_reverse_on_basic_element_wise_functions(self):
        cg = CGraph()
        D,P,N,M = 2,5,7,11
        ax = UTPM(numpy.random.rand(D,P,N,M))
        ay = UTPM(numpy.random.rand(D,P,N,M))
        fx = Function(ax)
        fy = Function(ay)
        fv1 = fx * fy
        fv2 = (fv1 * fx + fy)*fv1
        cg.independentFunctionList = [fx,fy]
        cg.dependentFunctionList = [fv2]

        v2bar = UTPM(numpy.zeros((D,P,N,M)))
        v2bar.tc[0,:,:,:] = 1.
        cg.reverse([v2bar])

        xbar_reverse = cg.independentFunctionList[0].xbar
        ybar_reverse = cg.independentFunctionList[1].xbar
        
        xbar_symbolic = 3. * ax*ax * ay*ay + ay*ay
        ybar_symbolic = 2.*ax*ax*ax * ay + 2. * ax * ay

        # print xbar_symbolic.tc
        # print xbar_reverse
        # print ybar_symbolic
        # print ybar_reverse
        
        assert_array_almost_equal(xbar_reverse.tc, xbar_symbolic.tc)
        assert_array_almost_equal(ybar_reverse.tc, ybar_symbolic.tc)

    def test_dot(self):
        cg = CGraph()
        D,P,N,M = 2,5,7,11
        ax = UTPM(numpy.random.rand(D,P,N,M))
        ay = UTPM(numpy.random.rand(D,P,M,N))
        fx = Function(ax)
        fy = Function(ay)
        fz = fx.dot(fy)
        cg.independentFunctionList = [fx,fy]
        cg.dependentFunctionList = [fz]
        
        ax = UTPM(numpy.random.rand(D,P,N,M))
        ay = UTPM(numpy.random.rand(D,P,M,N))
        azbar = UTPM(numpy.random.rand(*fz.x.tc.shape))
        cg.forward([ax,ay])
        cg.reverse([azbar])
        
        xbar_reverse = cg.independentFunctionList[0].xbar
        ybar_reverse = cg.independentFunctionList[1].xbar        
        
        xbar_symbolic = azbar.dot(ay.T)
        ybar_symbolic = (ax.T).dot(azbar)
        
        assert_array_almost_equal(xbar_reverse.tc, xbar_symbolic.tc)
        assert_array_almost_equal(ybar_reverse.tc, ybar_symbolic.tc)
        
        
    def test_transpose(self):
        cg = CGraph()
        D,P,N,M = 2,5,7,11
        ax = UTPM(numpy.random.rand(D,P,N,M))
        fx = Function(ax)
        fy = fx.T
        cg.independentFunctionList = [fx]
        cg.dependentFunctionList = [fy]

        assert_array_equal(fy.shape, (M,N))
        assert_array_equal(fy.x.tc.shape, (D,P,M,N))
        
        cg.forward([ax])
        assert_array_equal(cg.dependentFunctionList[0].shape, (M,N))
        assert_array_equal(cg.dependentFunctionList[0].x.tc.shape, (D,P,M,N))
    
    def test_part_of_ODOE_objective_function(self):
        D,P,N,M = 2,5,100,3
        MJs = [ UTPM(numpy.random.rand(D,P,N,M)), UTPM(numpy.random.rand(D,P,N,M))]
        cg = CGraph()
        FJs = [Function(MJ) for MJ in MJs]
        FPhi = numpy.sum([ (FJ.T).dot(FJ) for FJ in FJs ])
        cg.independentFunctionList = FJs
        cg.dependentFunctionList = [FPhi]
        
        assert_array_equal(FPhi.shape, (M,M))
        cg.forward(MJs)
        assert_array_equal(cg.dependentFunctionList[0].x.tc.shape, [D,P,M,M])


class TestCGraphOnCTPS(TestCase):
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
        cg.forward([ax,ay])
        assert_array_almost_equal(cg.dependentFunctionList[0].x.tc, ((ax*ay * ax + ay)*ax*ay).tc)




        
if __name__ == "__main__":
    run_module_suite()




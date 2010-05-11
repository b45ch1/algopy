from numpy.testing import *
import os

from algopy.tracer.tracer import *
from algopy.utpm import UTPM
from algopy.utps import UTPS
from algopy.ctps import CTPS

import numpy

# class TestInverse(TestCase):
#     X = UTPM(numpy.array([[[[2.]]]]))
#     Y  = UTPM.inv(X)
#     Ybar = UTPM(numpy.array([[[[1.]]]]))
#     Xbar = UTPM.pb_inv(Ybar, X, Y)
    
#     print X,Y
#     print Xbar

# class TestCGraphOnSclars(TestCase):
#     def test_reverse(self):
#         cg = CGraph()
#         x = 3.
#         y = 7.
#         fx = Function(x)
#         fy = Function(y)
#         fv1 = fx * fy
#         fv2 = (fv1 * fx + fy)*fv1
#         cg.independentFunctionList = [fx,fy]
#         cg.dependentFunctionList = [fv2]

#         v2bar = 1.
#         cg.reverse([v2bar])

#         xbar_reverse = cg.independentFunctionList[0].xbar
#         ybar_reverse = cg.independentFunctionList[1].xbar

#         xbar_symbolic = 3 * x**2 * y**2 + y**2
#         ybar_symbolic = 2*x**3 * y + 2 * x * y

#         assert_almost_equal(xbar_reverse, xbar_symbolic)
#         assert_almost_equal(ybar_reverse, ybar_symbolic)


# class TestCGraphOnUTPS(TestCase):
#     def test_reverse(self):
#         cg = CGraph()
#         ax = UTPS([3.,1.])
#         ay = UTPS([7.,0.])
#         fx = Function(ax)
#         fy = Function(ay)
#         fv1 = fx * fy
#         fv2 = (fv1 * fx + fy)*fv1
#         cg.independentFunctionList = [fx,fy]
#         cg.dependentFunctionList = [fv2]

#         v2bar = UTPS([1.,0.])
#         cg.reverse([v2bar])

#         xbar_reverse = cg.independentFunctionList[0].xbar
#         ybar_reverse = cg.independentFunctionList[1].xbar
        
#         xbar_symbolic = 3 * ax**2 * ay**2 + ay**2
#         ybar_symbolic = 2*ax**3 * ay + 2 * ax * ay

#         # print xbar_symbolic.tc
#         # print xbar_reverse
#         # print ybar_symbolic
#         # print ybar_reverse
        
#         assert_array_almost_equal(xbar_reverse.tc, xbar_symbolic.tc)
#         assert_array_almost_equal(ybar_reverse.tc, ybar_symbolic.tc)


# class TestCGraphOnUTPM(TestCase):

#     def test_plotting_the_graph(self):
#         import os.path
#         cg = CGraph()
#         X = 2 * numpy.random.rand(2,2,2,2)
#         Y = 2 * numpy.random.rand(2,2,2,2)
#         AX = UTPM(X)
#         AY = UTPM(Y)
#         FX = Function(AX)
#         FY = Function(AY)

#         FX = FX*FY
#         FX = FX.dot(FY) + FX.transpose()
#         FX = FY + FX * FY
#         FY = FX.inv()
#         FY = FY.transpose()
#         FZ = FX * FY
#         FW = Function([[FX, FZ], [FZ, FY]])
#         FTR = FW.trace()
#         cg.independentFunctionList = [FX, FY]
#         cg.dependentFunctionList = [FTR]
#         cg.plot(filename = '/tmp/cgraph_matrix.png', method = None, orientation = 'TD')
#         assert os.path.isfile('/tmp/cgraph_matrix.png')
#         #os.remove('/tmp/cgraph.png')



# class TestCGraphOnCTPS(TestCase):
#     def test_reverse(self):
#         cg = CGraph()
#         ax = CTPS_C([3.,1.,0.,0.])
#         ay = CTPS_C([7.,0.,0.,0.])
#         fx = Function(ax)
#         fy = Function(ay)
#         fv1 = fx * fy
#         fv2 = (fv1 * fx + fy)*fv1
#         cg.independentFunctionList = [fx,fy]
#         cg.dependentFunctionList = [fv2]
#         cg.forward([ax,ay])
        
#         v2bar = CTPS_C([1.,0.,0.,0.])
#         cg.reverse([v2bar])

#         xbar_reverse = cg.independentFunctionList[0].xbar
#         ybar_reverse = cg.independentFunctionList[1].xbar
        
#         xbar_symbolic = 3 * ax*ax * ay*ay + ay*ay
#         ybar_symbolic = 2*ax*ax*ax * ay + 2 * ax * ay

#         # print xbar_symbolic.data
#         # print xbar_reverse.data
#         # # print ybar_symbolic
#         # # print ybar_reverse
        
#         assert_array_almost_equal(xbar_reverse.data, xbar_symbolic.data)
#         assert_array_almost_equal(ybar_reverse.data, ybar_symbolic.data)









if __name__ == "__main__":
    run_module_suite()

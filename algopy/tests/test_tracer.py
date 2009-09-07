from numpy.testing import *
import numpy

from algopy.tracer import *


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
        

class TestCGraph(TestCase):
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


if __name__ == "__main__":
    run_module_suite()




from numpy.testing import *
import numpy

from algopy.utp.utps import *


class ElementaryFunctions(TestCase):
    def test_UTPS(self):
        """
        this checks _only_ if calling the operations is ok
        """
        P = 3
        D = 7
        X = 2 * numpy.random.rand(D,P)
        Y = 3 * numpy.random.rand(D,P)

        AX = UTPS(X)
        AY = UTPS(Y)
        AZ = AX + AY
        AZ = AX - AY
        AZ = AX * AY
        AZ = AX / AY

    def test_simple_multipication(self):
        """differentiation of f(x,y) = x*y at [5,7] in direction [13,17]"""
        def f(z):
            return z[0]*z[1]
        a = numpy.array([UTPS([5.,13.]),UTPS([7.,17.])])
        assert_array_almost_equal([[35.] ,[176.]], f(a).tc)

    def test_abs(self):
        """differentiation of abs(x) at x=-2.3"""
        def f(x):
            return numpy.abs(x)
        a = UTPS(numpy.array([-2.3,1.3]))
        b = f(a)
        correct_result = UTPS(numpy.array([2.3,-1.3]))
        assert_array_almost_equal(correct_result.tc,b.tc)
        a = UTPS(numpy.array([5.1,1.]))
        b = f(a)
        correct_result = UTPS(numpy.array([5.1,1.]))
        assert_array_almost_equal(correct_result.tc, b.tc)


    def test_sqrt(self):
        """\ndirectional derivative of sqrt(x) at x = 3.1 with direction d = 7.4"""
        def f(x):
            return numpy.sqrt(x)
        a = UTPS([3.1,7.4])
        b = f(a)
        correct_result = UTPS( [numpy.sqrt(3.1), 0.5/numpy.sqrt(3.1) * 7.4 ])
        assert_array_almost_equal(correct_result.tc, b.tc)

    def test_mixed_arguments_double_UTPS(self):
        """\nfunction f(ax,y) = ax+y that works on mixed arguments, i.e. doubles (y) and UTPS (x)"""
        def f(x,y):
            return x+y
        a1 = UTPS([2.,13.])
        a2 = 1.
        b = f(a1,a2)
        correct_result = UTPS([f(a1.tc[0,0],a2), 13.])
        assert_array_almost_equal(correct_result.tc, b.tc)

    def test_double_mul_UTPS(self):
        """ function f(ax,y) = ax * y"""
        def f(x,y):
            return x*y
        a1 = UTPS([[2.],[13.]])
        a2 = 5.
        b = f(a1,a2)
        correct_result = UTPS([[f(a1.tc[0,0],a2)], [65.]])
        assert_array_almost_equal(correct_result.tc, b.tc)

    def test_multivariate_function(self):
        """computing directional derivative of a function f:R^3 -> R^2 with direction d=[1,1,1]"""
        def f(x):
            return numpy.array([x[0]*x[1]*x[2], x[0]*x[0]*x[2]])
        def df(x,h):
            jac = numpy.array(	[[x[1]*x[2], 2* x[0]*x[2]],
                                [x[0]*x[2],0],
                                [x[0]*x[1], x[0]**2]])
            return numpy.dot(jac.T,h)

        a = [UTPS([1.,1.]),UTPS([2.,1.]),UTPS([3.,1.])]
        b = f(a)
        h = [1,1,1]
        fa = f([1.,2.,3.])
        dfh = df([1.,2.,3.],h)
        correct_result = numpy.array(
            [UTPS([fa[0],dfh[0]]),UTPS([fa[1],dfh[1]])])
        assert_array_almost_equal( correct_result[0].tc,b[0].tc)
        assert_array_almost_equal( correct_result[1].tc,b[1].tc)




class NumpyArrayOperationsTests(TestCase):
    def test_numpy_slicing(self):
        """ f= sum(x[1:]*x[-2::-1])"""
        def f(x):
            return numpy.sum(x[:]*x[-1::-1])
        def df(x,h):
            return numpy.dot(2*x[::-1],h)

        ax = numpy.array(
            [UTPS([1.,1.]),UTPS([2.,0]),UTPS([3.,0])]
            )
        ay = f(ax)
        x = numpy.array([1,2,3])
        h = [1,0,0]
        correct_result = UTPS(
            [f(x),df(x,h)]
            )
        assert_array_almost_equal(correct_result.tc, ay.tc)
        
    def test_numpy_linalg_norm(self):
        """\ndirectional derivative of norm(x) at x=[2.1,3.4] with direction d = [5.6,7.8]"""
        def f(x):
            return numpy.linalg.norm(x)
        a = numpy.array([UTPS([2.1,5.6]),UTPS([3.4,7.8])])
        b = f(a)
        correct_result = UTPS([numpy.linalg.norm([2.1,3.4]), 9.57898451145 ])
        assert_array_almost_equal(correct_result.tc, b.tc)
        
    def test_qr_decomposition(self):
       N,D,P = 2,4,2
       A = numpy.array([[UTPS(numpy.random.rand(D,P)) for c in range(N)] for r in range(N)])
       
       Q,R =  qr(A)
       
   


if __name__ == "__main__":
    run_module_suite()


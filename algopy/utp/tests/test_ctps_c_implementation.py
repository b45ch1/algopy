from numpy.testing import *
import numpy

from algopy.utp.ctps_c_implementation import *

class Test_CTPS_C_operations(TestCase):
    def test_add(self):
        x1 = numpy.random.rand()
        x2 = numpy.random.rand()
    
        ax1 = CTPS_C(numpy.array([x1,1,0,0],dtype=float))
        ax2 = CTPS_C(numpy.array([x2,0,0,0],dtype=float))

        ay = ax1 + ax2
        assert_array_almost_equal(ay.data, ax1.data + ax2.data)
        
    def test_sub(self):
        x1 = numpy.random.rand()
        x2 = numpy.random.rand()
    
        ax1 = CTPS_C(numpy.array([x1,1,0,0],dtype=float))
        ax2 = CTPS_C(numpy.array([x2,0,0,0],dtype=float))

        ay = ax1 - ax2
        assert_array_almost_equal(ay.data, ax1.data - ax2.data)        

    def test_mul(self):
        x1 = numpy.random.rand()
        x2 = numpy.random.rand()

        ax1 = CTPS_C(numpy.array([x1,1,0,0],dtype=float))
        ax2 = CTPS_C(numpy.array([x2,0,2,0],dtype=float))

        ay = ax1 * ax2
        assert_array_almost_equal([x1*x2, x2, 2*x1, 2], ay.data)
        
    def test_div(self):
        x1 = numpy.random.rand()
        x2 = numpy.random.rand()

        ax1 = CTPS_C(numpy.array([x1,1,0,0],dtype=float))
        ax2 = CTPS_C(numpy.array([x2,0,2,0],dtype=float))

        ay = ax1 / ax2
        assert_array_almost_equal([x1/x2, 1./x2, -2*x1/x2**2, -2./x2**2], ay.data)        

    def test_simple_hessian(self):
        """
        test function:
        f: R^4 -> R
        x -> y = f(x) = prod(x)
    
        u = x[:2]
        v = x[2:]
    
        goal: Compute  d/du d/dv f(u,v)
    
    
        """
        x1 = numpy.random.rand()
        x2 = numpy.random.rand()
        x3 = numpy.random.rand()
        x4 = numpy.random.rand()
    
        # compute d/dx1 d/x3 f
        ax1 = CTPS_C(numpy.array([x1,1,0,0],dtype=float))
        ax2 = CTPS_C(numpy.array([x2,0,0,0],dtype=float))
        ax3 = CTPS_C(numpy.array([x3,0,1,0],dtype=float))
        ax4 = CTPS_C(numpy.array([x4,0,0,0],dtype=float))
    
        ay = ax1 * ax2 * ax3 * ax4
   
        assert_almost_equal(ay.data[2**2 - 1], x2*x4, decimal = 3)
        
        

if __name__ == "__main__":
    run_module_suite()         

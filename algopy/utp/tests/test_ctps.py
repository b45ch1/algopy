from numpy.testing import *
import numpy

from algopy.utp.ctps import *

class TestCTPSUtils(TestCase):
    def test_i2m(self):
        i = numpy.ones(10,dtype=bool)
        j = numpy.ones(10,dtype=bool)
        k = numpy.array([0,1,0,1],dtype=bool)
        assert i2m(i) == 2**10 -1
        assert i2m(i-j) == 0
        assert i2m(k) == 0 + 1*2 + 0*4 + 1*8 
    
    def test_m2i(self):
        m = 8
        i = m2i(m)
        assert_array_equal(i,[False, False, False,  True])
    
    def test_i2m_bits(self):
        m = 8
        bits = numpy.array([0,1,1,1,0,1,0],dtype=bool)
        i = m2i(m, bits)
        assert_array_equal(i,[False,  False, False,  False, False, True, False])
    
    def test_i2m_m2i_i2m_m2i(self):
        N = 20
        i1 = numpy.array([True] + list(numpy.random.randint(0,2,N)) + [True], dtype=bool)
        m1 = i2m(i1)
        i2 = m2i(m1)
        m2 = i2m(i2)
    
        assert_array_equal(i1,i2)
        assert_equal(m1,m2)
    
    def test_multi_index_sum(self):
        N = 3
        i = numpy.array([True] + list(numpy.random.randint(0,2,N)) + [True], dtype=bool)
        M = 2**numpy.sum(i)-1
        j = m2i(M,i)
        assert_array_equal(i,j)
        
    # def test_memory_access(self):
        # i = numpy.array([0,0,1,0,0,1], dtype=bool)
        # M = 2**numpy.sum(i)
        # for m in range(M):
            # print i2m(m2i(m,i))


class Test_CTPS_operations(TestCase):
    
    def test_inconv2(self):
        x = numpy.random.rand(8)
        y = numpy.random.rand(8)
        z = numpy.zeros(8)
        inconv2(z, x, y)
    
        # (0,0,0)
        assert_almost_equal(z[0], x[0]*y[0])
        # (1,0,0)
        assert_almost_equal(z[1], x[0]*y[1] + x[1]*y[0])
        # (1,1,0)
        assert_almost_equal(z[2], x[0]*y[2] + x[2]*y[0])
        assert_almost_equal(z[3], x[0]*y[3] + x[1]*y[2] + x[2]*y[1] + x[3]*y[0])
        # (1,1,1)
        assert_almost_equal(z[4], x[0]*y[4] + x[4]*y[0])
        assert_almost_equal(z[5], x[0]*y[5] + x[1]*y[4] + x[4]*y[1] + x[5]*y[0])
        assert_almost_equal(z[6], x[0]*y[6] + x[2]*y[4] + x[4]*y[2] + x[6]*y[0])
        assert_almost_equal(z[7], x[0]*y[7] + x[1]*y[6] + x[2]*y[5] + x[3]*y[4] + x[4]*y[3] + x[5]*y[2] + x[6]*y[1] + x[7]*y[0])
    
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
        ax1 = CTPS(numpy.array([x1,1,0,0],dtype=float))
        ax2 = CTPS(numpy.array([x2,0,0,0],dtype=float))
        ax3 = CTPS(numpy.array([x3,0,1,0],dtype=float))
        ax4 = CTPS(numpy.array([x4,0,0,0],dtype=float))
    
        ay = ax1 * ax2 * ax3 * ax4
   
        assert_almost_equal(ay.data[2**2 - 1], x2*x4, decimal = 3)
        
        

if __name__ == "__main__":
    run_module_suite()         

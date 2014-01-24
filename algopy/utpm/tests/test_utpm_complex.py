import math

from numpy.testing import *
import numpy
import scipy.special

import algopy.nthderiv
from algopy.utpm import *
from algopy import zeros

try:
    import mpmath
except ImportError:
    mpmath = None


class Test_Forward_Complex(TestCase):

    def test_add(self):
        D,P = 3,1
        x = UTPM(numpy.zeros((D,P), dtype='complex'))
        y = UTPM(numpy.zeros((D,P), dtype='complex'))

        x.data[:,0] = [1+1j,2+2j,3+3j]
        y.data[:,0] = [4+4j,5+5j,6+6j]

        z = UTPM(numpy.zeros((D,P), dtype='complex'))
        z.data[0,0] = 5+5j
        z.data[1,0] = 7+7j
        z.data[2,0] = 9+9j

        z1 = y + x
        z2 = x + y

        assert_array_almost_equal(z.data, z1.data)
        assert_array_almost_equal(z.data, z2.data)

    def test_sub(self):
        D,P = 3,1
        x = UTPM(numpy.zeros((D,P), dtype='complex'))
        y = UTPM(numpy.zeros((D,P), dtype='complex'))

        x.data[:,0] = [1+1j,2+2j,3+3j]
        y.data[:,0] = [4+4j,5+5j,6+6j]

        z = UTPM(numpy.zeros((D,P), dtype='complex'))
        z.data[0,0] = 3+3j
        z.data[1,0] = 3+3j
        z.data[2,0] = 3+3j

        z1 = y - x
        assert_array_almost_equal(z.data, z1.data)


    def test_mul(self):
        D,P = 3,1
        x = UTPM(numpy.zeros((D,P), dtype='complex'))
        y = UTPM(numpy.zeros((D,P), dtype='complex'))

        x.data[:,0] = [1+1j,2+2j,3+3j]
        y.data[:,0] = [4+4j,5+5j,6+6j]

        z = UTPM(numpy.zeros((D,P), dtype='complex'))
        z.data[0,0] = 4*2j
        z.data[1,0] = (5+8)*2j
        z.data[2,0] = (6 + 10 + 12)*2j

        z1 = y * x
        z2 = x * y
        assert_array_almost_equal(z.data, z1.data)
        assert_array_almost_equal(z.data, z2.data)

    def test_div(self):
        D,P = 3,1
        x = UTPM(numpy.zeros((D,P), dtype='complex'))
        y = UTPM(numpy.zeros((D,P), dtype='complex'))

        x.data[...] = numpy.random.random((D,P)) + 1j*numpy.random.random((D,P))
        y.data[...] = numpy.random.random((D,P)) + 1j*numpy.random.random((D,P))

        z = y / x
        u = z * x

        assert_array_almost_equal(y.data, u.data)

    def test_transpose(self):
        D,P,M = 3,1,4
        A = UTPM(numpy.zeros((D,P,M,M), dtype='complex'))
        A.data[...] = numpy.random.random((D,P,M,M)) + 1j*numpy.random.random((D,P,M,M))

        B = A.T
        for i in range(M):
            for j in range(M):
                assert_array_almost_equal(A[i,j].data, B[j,i].data)

if __name__ == "__main__":
    run_module_suite()

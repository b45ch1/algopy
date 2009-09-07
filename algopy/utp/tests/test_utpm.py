from numpy.testing import *
import numpy

from algopy.utp.utpm import *


class TestMatPoly(TestCase):
    def test_UTPM(self):
        """
        this checks _only_ if calling the operations is ok
        """
        X = 2 * numpy.random.rand(2,2,2,2)
        Y = 3 * numpy.random.rand(2,2,2,2)

        AX = MatPoly(X)
        AY = MatPoly(Y)
        AZ = AX + AY
        AZ = AX - AY
        AZ = AX * AY
        AZ = AX / AY
        AZ = AX.dot(AY)
        AZ = AX.inv()
        AZ = AX.trace()
        AZ = AX[0,0]
        AZ = AX.T
        AX = AX.set_zero()

    def test_trace(self):
        N1 = 2
        N2 = 3
        N3 = 4
        N4 = 5
        x = numpy.asarray(range(N1*N2*N3*N4))
        x = x.reshape((N1,N2,N3,N4))
        AX = MatPoly(x)
        AY = AX.T
        AY.TC[0,0,2,0] = 1234
        assert AX.TC[0,0,0,2] == AY.TC[0,0,2,0]




if __name__ == "__main__":
    run_module_suite()

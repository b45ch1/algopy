from numpy.testing import *
import numpy
import numpy.random

from algopy.utpm import UTPM
from algopy.utps import UTPS
from algopy.utils import *

class TestUtils ( TestCase ):
    def test_utpm2utps(self):
        x = UTPM( numpy.zeros((2,5,3,7),dtype=float))
        y = utpm2utps(x)
        
        assert_array_equal(y.shape, (3,7))

    def test_utps2utpm(self):
        D,P,N,M = 2,5,3,7
        x = numpy.array([[UTPS(numpy.random.rand(D,P)) for m in range(M)] for n in range(N)])
        y = utps2utpm(x)
        
        assert_array_equal(y.tc.shape, (D,P,N,M))
        
        
    def test_utps2base_and_dirs2utps(self):
        N,P,D = 2,5,3
        x = numpy.array([UTPS(numpy.random.rand(D,P))  for n in range(N)])
        y,W = utps2base_and_dirs(x)
        x2= base_and_dirs2utps(y,W)
        y2,W2= utps2base_and_dirs(x2)
        
        assert_array_almost_equal(y,y2)
        assert_array_almost_equal(W,W2)
        
        
    def test_utpm2base_and_dirs(self):
        D,P,N,M,K = 2,3,4,5,6
        u = UTPM( numpy.arange(D*P*N*M*K).reshape((D,P,N,M,K)))
        x,V = utpm2base_and_dirs(u)
        assert_array_almost_equal(x, numpy.arange(N*M*K).reshape((N,M,K)))
        assert_array_almost_equal(V, numpy.arange(P*N*M*K,D*P*N*M*K).reshape((D-1,P,N,M,K)).transpose((2,3,4,1,0)))
        
        
    def test_utpm2dirs(self):
        D,P,N,M,K = 2,3,4,5,6
        u = UTPM( numpy.arange(D*P*N*M*K).reshape((D,P,N,M,K)))
        Vbar = utpm2dirs(u)
        assert_array_almost_equal(Vbar, numpy.arange(D*P*N*M*K).reshape((D,P,N,M,K)).transpose((2,3,4,1,0)) )
        
if __name__ == "__main__":
    run_module_suite()

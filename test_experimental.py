from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utp.utpm import *


class Test_Experimental(TestCase):
    
    # def test_pb_cholesky(self):
        # D,P,N = 2, 1, 3
        # tmp = numpy.random.rand(*(D,P,N,N))
        # A = UTPM(tmp)
        # A = UTPM.dot(A.T,A)

        # L = UTPM.cholesky(A)
        # Lbar = UTPM(numpy.random.rand(*(D,P,N,N)))
        # for r in range(N):
            # for c in range(N):
                # Lbar[r,c] *= (r>=c)
        
        # # print Lbar
        # # print L
        
        # Abar = UTPM.pb_cholesky(Lbar, A, L)
        # print Abar
        
        # for p in range(P):
            # Ab = Abar.data[0,p]
            # Ad = A.data[1,p]

            # Lb = Lbar.data[0,p]
            # Ld = L.data[1,p]
            # assert_almost_equal(numpy.trace(numpy.dot(Ab.T,Ad)), numpy.trace(numpy.dot(Lb.T,Ld) ))


    # def test_something(self):
        
    #     D,P,N = 3, 1, 3
    #     tmp = numpy.zeros((D,P,N,N))
    #     tmp[0,0] = [[0,0,0],[0,0,0],[0,1,0]]
    #     A = UTPM(tmp)
    #     print A
    pass
        

if __name__ == "__main__":
    # run_module_suite() 
    from adolc import adouble
    from adolc.cgraph import AdolcProgram
    from adolc.linalg import qr
    D,P,N = 2, 1, 3
    
    A_data = numpy.random.rand(D,P,N,N)
    A_data[0,0] = numpy.array([[1.,0,0],[0,1.,-1],[0,-1,1.]])
    
    #----------------------------------------------
    # STEP 1:
    # QR decomposition by Givens Rotations
    # using pyadolc for the differentiation
    #----------------------------------------------
    A = A_data[0,0,:,:]
    
    # trace QR decomposition with adolc
    AP = AdolcProgram()
    AP.trace_on(1)
    aA = adouble(A)
    AP.independent(aA)
    aQ, aR = qr(aA)
    AP.dependent(aQ)
    AP.dependent(aR)
    AP.trace_off()
    
    # compute push forward
    VA = A_data[1:,...].transpose((2,3,1,0))
    # print VA
    out = AP.forward([A],[VA])
    
    Q_data = numpy.zeros((D,P,N,N))
    R_data = numpy.zeros((D,P,N,N))
    
    Q_data[0,0] = out[0][0]
    R_data[0,0] = out[0][1]
    
    Q_data[1:,0] = out[1][0][:,:,0,:].transpose((2,0,1))
    R_data[1:,0] = out[1][1][:,:,0,:].transpose((2,0,1))
    
    Q = UTPM(Q_data)
    R = UTPM(R_data)
    
    A = UTPM(A_data)
    print A - UTPM.dot(Q,R)
    
    print numpy.rank(A_data[0,0])
    
    Q,R = UTPM.qr(A[:,:])
    
    
    print A - UTPM.dot(Q,R)

    
    
    
    

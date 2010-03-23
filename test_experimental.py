from numpy.testing import *
import numpy

from algopy.tracer.tracer import *
from algopy.utp.utpm import *


class Test_Experimental(TestCase):
    
    def test_eigh(self):
        D,P,N = 3,1,6
        A = UTPM(numpy.zeros((D,P,N,N)))
        V = UTPM(numpy.random.rand(D,P,N,N))
        
        A.data[0,0] = numpy.diag([2,2,3,3.,4,5])
        A.data[1,0] = numpy.diag([5,1,3,1.,1,3])
        
        V,Rtilde = UTPM.qr(V)
        A = UTPM.dot(UTPM.dot(V.T, A), V)

        l,Q = UTPM.eigh(A)
        L = UTPM.diag(l)
        
        print l
        
        
        
        # D,P,N = 3,1,6
        # A = UTPM(numpy.zeros((D,P,N,N)))
        # V = UTPM(numpy.random.rand(D,P,N,N))
        
        # A.data[0,0] = numpy.diag([2,2,2,3,3,4])
        # A.data[1,0] = numpy.diag([1,1,2,2,2,5])
        # A.data[2,0] = numpy.diag([1,1,1,7,3,1])
        
        # V,Rtilde = UTPM.qr(V)
        # A = UTPM.dot(UTPM.dot(V.T, A), V)
        
        # l,Q = UTPM.eigh(A)
        
        # # print Q
        # # print UTPM.dot(Q.T,Q)
        
        # # print A
        # print UTPM.dot(Q.T, UTPM.dot(A, Q))
    
    # def test_q_lift(self):
    #     from algopy.utp.utpm.algorithms import vdot
    #     d,D,P,N = 2,4,5,7
    #     A = numpy.zeros((D,P,N,N))
    #     A[0,:] = numpy.random.rand(P,N,N)
    #     A[1,:] = numpy.random.rand(P,N,N)
    #     A = UTPM(A)
        
    #     Q = UTPM.qr(A)[0]
    #     Q.data[2,...] = 0
        
    #     print Q
    #     Q = Q.data
        
    #     def lift_Q(Q, d, D):
    #         S = numpy.zeros((P,N,N))
    #         for k in range(d,D):
    #             S *= 0
    #             for i in range(1,k):
    #                 S += vdot(Q[i,...].transpose(0,2,1), Q[k-i,...])
                
    #             for p in range(P):
    #                 Q[k,p] = -0.5 * numpy.dot(Q[0,p], S[p])
                    
    #         return Q
        
    #     Q = lift_Q(Q,d,D)
        
    #     Q = UTPM(Q)
        
    #     # print Q
        
    #     print UTPM.dot(Q.T,Q)
                
        
        # Q_tmp = numpy.zeros((DT,  stop-start, stop-start))
        # Q_tmp[:DT-D] = Q_hat_data
        # print Q_tmp
        
        # dG = numpy.zeros((stop-start, stop-start))
        # for k in range(D,DT):
        #     dG *= 0
        #     for i in range(1,k):
        #         dG += numpy.dot(Q_tmp[i].T, Q_tmp[k-i])
        #         Q_tmp[k] = -0.5 * numpy.dot(Q_tmp[0].T, dG)
        
        # Q_tmp = Q_tmp.reshape((DT,1, stop-start, stop-start))
        
        
    
    # def test_numerical_stability_qr(self):
        # from adolc import adouble
        # from adolc.cgraph import AdolcProgram
        # from adolc.linalg import qr
        # D,P,N = 2, 1, 3
        
        # A_data = numpy.random.rand(D,P,N,N)
        # A_data[0,0] = numpy.array([[1.,0,0],[0,1.,-1],[0,-1,1.]])
        
        # #----------------------------------------------
        # # STEP 1:
        # # QR decomposition by Givens Rotations
        # # using pyadolc for the differentiation
        # #----------------------------------------------
        # A = A_data[0,0,:,:]
        
        # # trace QR decomposition with adolc
        # AP = AdolcProgram()
        # AP.trace_on(1)
        # aA = adouble(A)
        # AP.independent(aA)
        # aQ, aR = qr(aA)
        # AP.dependent(aQ)
        # AP.dependent(aR)
        # AP.trace_off()
        
        # # compute push forward
        # VA = A_data[1:,...].transpose((2,3,1,0))
        # # print VA
        # out = AP.forward([A],[VA])
        
        # Q_data = numpy.zeros((D,P,N,N))
        # R_data = numpy.zeros((D,P,N,N))
        
        # Q_data[0,0] = out[0][0]
        # R_data[0,0] = out[0][1]
        
        # Q_data[1:,0] = out[1][0][:,:,0,:].transpose((2,0,1))
        # R_data[1:,0] = out[1][1][:,:,0,:].transpose((2,0,1))
        
        # Q = UTPM(Q_data)
        # R = UTPM(R_data)
        
        # A = UTPM(A_data)
        # print A - UTPM.dot(Q,R)
        
        # print numpy.rank(A_data[0,0])
        
        # Q,R = UTPM.qr(A[:,:])
        
        
        # print A - UTPM.dot(Q,R)
    
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
    run_module_suite() 


    
    
    
    

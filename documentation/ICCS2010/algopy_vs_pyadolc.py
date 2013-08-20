from adolc import adouble
from adolc.cgraph import AdolcProgram
from adolc.linalg import qr
import numpy
import numpy.testing
from algopy.utpm import UTPM
from time import time
import os

repetitions = 10
D_list = [4]
N_list = [2**i for i in range(0,8)]
P_list = [1]

runtime_ratios_push_forward = numpy.zeros(( len(D_list), len(P_list), len(N_list), repetitions),dtype=float)

runtime_ratios_pullback = runtime_ratios_push_forward.copy()



for np,P in enumerate(P_list):
    for nn,N in enumerate(N_list):
        for nd,D in enumerate(D_list):

            print('running runtime tests for A.shape = (D,P,N,N) = %d, %d, %d, %d'%(D,P,N,N))
            A_data = numpy.random.rand(N,N,P,D)
            Qbar_data = numpy.random.rand(1,N,N,P,D)
            Rbar_data = numpy.random.rand(1,N,N,P,D)

            #----------------------------------------------
            # STEP 1:
            # QR decomposition by Givens Rotations
            # using pyadolc for the differentiation
            #----------------------------------------------
            A = A_data[:,:,0,0]
            # trace QR decomposition with adolc
            AP = AdolcProgram()
            AP.trace_on(1)
            aA = adouble(A)
            AP.independent(aA)
            aQ, aR = qr(aA)
            AP.dependent(aQ)
            AP.dependent(aR)
            AP.trace_off()

            for r in range(repetitions):

                A = A_data[:,:,0,0]
                # compute push forward
                VA = A_data[:,:,:,1:]
                tic = time()
                out = AP.forward([A],[VA])
                toc = time()
                runtime_pyadolc_push_forward = toc - tic

                adolc_Q = out[0][0]
                adolc_R = out[0][1]
                adolc_VQ = out[1][0]
                adolc_VR = out[1][1]

                # compute pullback
                WQ = Qbar_data
                WR = Rbar_data
                tic = time()
                out = AP.reverse([WQ, WR])
                toc = time()
                runtime_pyadolc_pullback = toc - tic

                #----------------------------------------------
                # STEP 2:
                # QR decomposition using LAPACK
                # using algopy for the differentiation
                #----------------------------------------------

                # comute push forward
                A = UTPM(numpy.ascontiguousarray(A_data.transpose((3,2,0,1))))
                Q = UTPM(numpy.zeros((D,P,N,N)))
                R = UTPM(numpy.zeros((D,P,N,N)))
                tic = time()
                Q,R = UTPM.qr(A, out = (Q,R))
                toc = time()
                runtime_algopy_push_forward = toc - tic

                # compute pullback
                Qbar = UTPM(numpy.ascontiguousarray(Qbar_data[0,...].transpose((3,2,0,1))))
                Rbar = UTPM(numpy.ascontiguousarray(Rbar_data[0,...].transpose((3,2,0,1))))
                tic = time()
                Q,R = UTPM.qr(A)
                Abar = UTPM.pb_qr(Qbar, Rbar, A, Q, R)
                toc = time()
                runtime_algopy_pullback = toc - tic

                push_forward_ratio = runtime_algopy_push_forward/runtime_pyadolc_push_forward
                pullback_ratio = runtime_algopy_pullback/runtime_pyadolc_pullback
                print('relative runtime of the push forward: algopy/pyadolc =', push_forward_ratio)

                print('relative runtime of the pullback: algopy/pyadolc =',pullback_ratio)

                runtime_ratios_push_forward[nd,np,nn,r] = push_forward_ratio
                runtime_ratios_pullback[nd,np,nn,r] = pullback_ratio

# Plot runtime ratios
from prettyplotting import *

# import pylab
pylab.figure()
pylab.title('Runtime Comparison QR-decomposition Push Forward ALGOPY vs PYADOLC')

stds =  numpy.std(runtime_ratios_push_forward[0,0,:,:],axis=1)
ms = numpy.mean(runtime_ratios_push_forward[0,0,:,:],axis=1)
pylab.errorbar(N_list, ms , yerr = stds )
pylab.loglog([],[])
#pylab.loglog(N_list, numpy.mean(runtime_ratios_push_forward[0,0,:,:],axis=1))
pylab.xlabel(r'$N$')
pylab.ylabel(r'runtime ratio algopy/pyadolc ')
pylab.grid()
pylab.savefig(os.path.join(os.path.dirname(__file__),'algopy_vs_pyadolc_push_forward.eps'))
# pylab.savefig(os.path.join(os.path.dirname(__file__),'algopy_vs_pyadolc_push_forward.png'))

pylab.figure()
pylab.title('Runtime Comparison QR-decomposition Pullback ALGOPY vs PYADOLC')
stds =  numpy.std(runtime_ratios_pullback[0,0,:,:],axis=1)
ms = numpy.mean(runtime_ratios_pullback[0,0,:,:],axis=1)
pylab.errorbar(N_list, ms , yerr = stds )
pylab.loglog([],[])
pylab.xlabel(r'$N$')
pylab.ylabel(r'runtime ratio algopy/pyadolc ')
pylab.grid()
pylab.savefig(os.path.join(os.path.dirname(__file__),'algopy_vs_pyadolc_pullback.eps'))
# pylab.savefig(os.path.join(os.path.dirname(__file__),'algopy_vs_pyadolc_pullback.png'))


pylab.show()






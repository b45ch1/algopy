"""
Experimentally test how stable the QR decomposition is.

Low rank matrices A are built by low rank updates, then it is tested how big the
residual R = QR - A  is.
"""

from numpy.testing import *
import numpy
import matplotlib.pyplot as pyplot
import prettyplotting

# pyplot.rcParams.update({'figure.figsize': (6,3.5)}) 

from algopy.tracer.tracer import *
from algopy.utpm import *



def create_matrix(D,P,N, a_list):
    """
    create low rank update matrices,
    
    A =  \sum_{i=0}^Na a_i x_i x_i^T
    
    where a is a list of weights of length Na,
    x_i are randomly generated vectors
    """
    
    for na, a in enumerate(a_list):
        x = UTPM(numpy.random.rand(D,P,N,1))
        if na == 0:
            A =  a * UTPM.dot(x,x.T)
        else:
            A += a * UTPM.dot(x,x.T)
            
    return A

D,P,N = 3,1,30
repetitions = 100
rank_list = [1,5,N]

residuals_dict = dict()
for rank in rank_list:
    a_list = numpy.ones(rank)
    residuals_dict[rank] = []
    
    for rep in range(repetitions):
        A = create_matrix(D,P,N,a_list)
        Q,R = UTPM.qr(A)
        
        tmp = []
        for d in range(D):
            tmp.append(numpy.abs((UTPM.dot(Q,R) - A).data[d]).max())
            
        residuals_dict[rank].append(tmp)
        
    residuals_dict[rank] = numpy.asarray(residuals_dict[rank])
        

for rank in rank_list:
    pyplot.figure()
    for d in range(D):
        pyplot.subplot(D,1,d+1)
        if d == 0:
            pyplot.title(r'max $ (| QR - A |)$, rank(A) = %d'%rank)
            
        tmp = numpy.sort(residuals_dict[rank].ravel())
        ran = tmp[0], tmp[(9*tmp.size)//10]
        print 'ran=',ran
        n, bins, patches = pyplot.hist(residuals_dict[rank][:,d], 20, histtype='bar', range = ran, label=r'degree $d=%d$'%d)
        pyplot.setp(patches, 'facecolor', '#444444')
        pyplot.legend( loc='best')
        pyplot.savefig('/tmp/qr_residuals_rank%d.eps'%rank)

pyplot.show()

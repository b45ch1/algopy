"""
Experimentally test how stable the QR decomposition is.

Low rank matrices A are built by low rank updates, then it is tested how big the
residuals res_1 = QR - A and res_2 = Q.T Q - I and res_3 = P_L ( R)

"""

from numpy.testing import *
import numpy
import matplotlib.pyplot as pyplot
import prettyplotting

from algopy import UTPM, qr, dot, triu

D,P,N = 3,1,3

x1 = UTPM(numpy.random.rand(D,P,N,1))
x2 = UTPM(numpy.random.rand(D,P,N,1))
x3 = UTPM(numpy.random.rand(D,P,N,1))

def alpha(beta):
    return numpy.array([1., 2.**(-beta/2.), 2.**(-beta)])

# create matrix A by lowrank updates, alpha triggeres what the rank is


res_1_list = []
res_2_list = []
res_3_list = []

betas = list(range(0,200,10))
for beta in betas:
    a = alpha(beta)
    A = a[0]*dot(x1,x1.T) + a[1]*dot(x2,x2.T) + a[2]*dot(x3,x3.T)

    Q,R = qr(A)
    res_1 = numpy.abs((dot(Q,R) - A).data).max()
    res_2 = numpy.abs((dot(Q.T,Q) - numpy.eye(N)).data).max()
    res_3 = numpy.abs((triu(R) - R).data).max()
    
    res_1_list.append(res_1)
    res_2_list.append(res_2)
    res_3_list.append(res_3)
      

pyplot.figure()
pyplot.semilogy(betas, res_1_list, 'kd', label=r'max $ (| QR - A |)$')
pyplot.semilogy(betas, res_2_list, 'ko', label=r'max $ (| Q^T Q - I |)$')
pyplot.semilogy(betas, res_3_list, 'k.', label=r'max $ (| P_R \circ R - R |)$')
pyplot.xlabel(r'$\beta$')


pyplot.legend( loc='best')
# #         pyplot.savefig('/tmp/qr_residuals_rank%d.eps'%rank)

pyplot.show()

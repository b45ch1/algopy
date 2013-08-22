"""
In this file, the numerical accuracy and the speed tests for QR and eigh functions
are performed.
"""

import numpy
import time
import algopy.utpm as utpm

# D-1 is the degree of the polynomial
# P is the number of simultaneously computed decompositions (Single Program Multiple Data)
# M is the number of rows of the matrix A
# N is the number of columsn of the matrix A


D,P,M,N = 5,4,100,5
print('')
print('-----------------------------------------------------------------------------------------------------------')
print('testing SPMD (%d datasets) differentiated QR decomposition for matrix A.shape = (%d,%d) up to %d\'th order'%(P,M,N,D-1))
print('-----------------------------------------------------------------------------------------------------------')
print('')

A = utpm.UTPM(numpy.random.rand(D,P,M,N))

tic = time.time()
Q,R = utpm.UTPM.qr(A)
toc = time.time()
runtime_push_forward = toc - tic

# check that normal QR decomposition matches the push forward version
for p in range(P):
    A0 = numpy.ascontiguousarray(A.data[0,p])
    Q0,R0 = numpy.linalg.qr(A0)
    numpy.testing.assert_array_almost_equal(Q.data[0,p],Q0)
    numpy.testing.assert_array_almost_equal(R.data[0,p],R0)

# check that the pushforward computed correctly
B = utpm.UTPM.dot(Q,R)
print('largest error for all QR decompositions and derivative degrees is: ',numpy.max(A.data - B.data))


# check runtime
runtime_normal = 0.
for p in range(P):
    A0 = numpy.ascontiguousarray(A.data[0,p])
    tic = time.time()
    Q0,R0 = numpy.linalg.qr(A0)
    toc = time.time()
    runtime_normal += toc - tic

print('measured runtime ratio push_forward/normal: ', runtime_push_forward/runtime_normal)







D,P,N = 5,5,20
print('')
print('-----------------------------------------------------------------------------------------------------------')
print('testing SPMD (%d datasets) push forward eig decomposition for matrix A.shape = (%d,%d) up to %d\'th order'%(P,N,N,D-1))
print('-----------------------------------------------------------------------------------------------------------')
print('')

# create symmetric matrix
A = utpm.UTPM(numpy.random.rand(D,P,N,N))
A = utpm.UTPM.dot(A.T,A)

tic = time.time()
l,Q = utpm.UTPM.eigh(A)
toc = time.time()
runtime_push_forward = toc - tic

# check that normal QR decomposition matches the push forward version
for p in range(P):
    A0 = numpy.ascontiguousarray(A.data[0,p])
    l0,Q0 = numpy.linalg.eigh(A0)
    numpy.testing.assert_array_almost_equal(Q.data[0,p],Q0)
    numpy.testing.assert_array_almost_equal(l.data[0,p],l0)

# check that the pushforward computed correctly
L = utpm.UTPM.diag(l)
B = utpm.UTPM.dot(utpm.UTPM.dot(Q,L), Q.T)
print('largest error for all eigenvalue decompositions and derivative degrees is: ',numpy.max(A.data - B.data))

# check runtime
runtime_normal = 0.

for p in range(P):
    A0 = numpy.ascontiguousarray(A.data[0,p])
    tic = time.time()
    Q0,R0 = numpy.linalg.eig(A0)
    toc = time.time()
    runtime_normal += toc - tic

print('measured runtime ratio push_forward/normal: ', runtime_push_forward/runtime_normal)








D,P,M,N = 2,1,50,8
print('')
print('--------------------------------------------------------------------------------------------------------------------')
print('testing SPMD (%d datasets) pullback qr decomposition for matrix A.shape = (%d,%d) up to first order'%(P,M,N))
print('--------------------------------------------------------------------------------------------------------------------')
print('')

# create symmetric matrix
A = utpm.UTPM(numpy.random.rand(D,P,M,N))

# compute push forward
tic = time.time()
Q,R = utpm.UTPM.qr(A)
toc = time.time()
runtime_push_forward = toc - tic

# compute pullback
Qbar = utpm.UTPM(numpy.random.rand(D,P,M,N))
Rbar = utpm.UTPM(numpy.random.rand(D,P,N,N))
tic = time.time()
Abar = utpm.UTPM.pb_qr(Qbar, Rbar, A, Q, R)
toc = time.time()
runtime_pullback = toc - tic

# check correctness of the pullback
# using the formula  < fbar, fdot > = < xbar, xdot>
# without using exact interpolation it is not possible to test the correctness of the pullback
# to higher order than one
Ab = Abar.data[0,0]
Ad = A.data[1,0]

Rb = Rbar.data[0,0]
Rd = R.data[1,0]

Qb = Qbar.data[0,0]
Qd = Q.data[1,0]

print(' (Abar, Adot) - (Rbar, Rdot) - (Qbar, Qdot) = %e'%( numpy.trace( numpy.dot(Ab.T, Ad)) - numpy.trace( numpy.dot(Rb.T, Rd) + numpy.dot(Qb.T, Qd))))








D,P,N = 2,1,8
print('')
print('--------------------------------------------------------------------------------------------------------------------')
print('testing SPMD (%d datasets) pullback eigenvalue decomposition for matrix A.shape = (%d,%d) up to first order'%(P,N,N))
print('--------------------------------------------------------------------------------------------------------------------')
print('')

# create symmetric matrix
A = utpm.UTPM(numpy.random.rand(D,P,N,N))
A = utpm.UTPM.dot(A.T,A)

# compute push forward
tic = time.time()
l,Q = utpm.UTPM.eigh(A)
toc = time.time()
runtime_push_forward = toc - tic

# compute pullback
Qbar = utpm.UTPM(numpy.random.rand(D,P,N,N))
lbar = utpm.UTPM(numpy.random.rand(D,P,N))
tic = time.time()
Abar = utpm.UTPM.pb_eigh( lbar, Qbar, A, l, Q)
toc = time.time()
runtime_pullback = toc - tic

# check correctness of the pullback
# using the formula  < fbar, fdot > = < xbar, xdot>
# without using exact interpolation it is not possible to test the correctness of the pullback
# to higher order than one
Ab = Abar.data[0,0]
Ad = A.data[1,0]

Lb = numpy.diag( lbar.data[0,0])
Ld = numpy.diag( l.data[1,0] )

Qb = Qbar.data[0,0]
Qd = Q.data[1,0]
print(' (Abar, Adot) - (Lbar, Ldot) - (Qbar, Qdot) = %e'%( numpy.trace( numpy.dot(Ab.T, Ad)) - numpy.trace( numpy.dot(Lb.T, Ld) + numpy.dot(Qb.T, Qd))))

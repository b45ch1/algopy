import numpy
import time
import algopy.utp.utpm as utpm

# D-1 is the degree of the polynomial
# P is the number of simultaneously computed decompositions (Single Program Multiple Data)
# M is the number of rows of the matrix A
# N is the number of columsn of the matrix A



# --------------------------------------------
# compute push forward of the QR decomposition
# --------------------------------------------
D,P,M,N = 5,5,100,5
print ''
print '-----------------------------------------------------------------------------------------------------------'
print 'testing SPMD (%d datasets) differentiated QR decomposition for matrix A.shape = (%d,%d) up to %d\'th order'%(P,M,N,D)
print '-----------------------------------------------------------------------------------------------------------'
print ''


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
print 'largest error for all QR decompositions and derivative degrees is: ',numpy.max(A.data - B.data)


# check runtime
tic = time.time()
for p in range(P):
    A0 = numpy.ascontiguousarray(A.data[0,p])
    Q0,R0 = numpy.linalg.qr(A0)
toc = time.time()
runtime_normal = toc - tic

print 'measured runtime ratio push_forward/normal: ', runtime_push_forward/runtime_normal


# ----------------------------------------------------
# compute push forward of the eigenvalue decomposition
# ----------------------------------------------------
D,P,N = 5,5,8
print ''
print '-----------------------------------------------------------------------------------------------------------'
print 'testing SPMD (%d datasets) differentiated eig decomposition for matrix A.shape = (%d,%d) up to %d\'th order'%(P,N,N,D)
print '-----------------------------------------------------------------------------------------------------------'
print ''

# create symmetric matrix
A = utpm.UTPM(numpy.random.rand(D,P,N,N))
A = utpm.UTPM.dot(A.T,A)

tic = time.time()
l,Q = utpm.UTPM.eig(A)
toc = time.time()
runtime_push_forward = toc - tic

# check that normal QR decomposition matches the push forward version
for p in range(P):
    A0 = numpy.ascontiguousarray(A.data[0,p])
    l0,Q0 = numpy.linalg.eig(A0)
    numpy.testing.assert_array_almost_equal(Q.data[0,p],Q0)
    numpy.testing.assert_array_almost_equal(l.data[0,p],l0)

# check that the pushforward computed correctly
L = utpm.UTPM.diag(l)
B = utpm.UTPM.dot(utpm.UTPM.dot(Q,L), Q.T)
print 'largest error for all eigenvalue decompositions and derivative degrees is: ',numpy.max(A.data - B.data)

# check runtime
tic = time.time()
for p in range(P):
    A0 = numpy.ascontiguousarray(A.data[0,p])
    Q0,R0 = numpy.linalg.eig(A0)
toc = time.time()
runtime_normal = toc - tic

print 'measured runtime ratio push_forward/normal: ', runtime_push_forward/runtime_normal

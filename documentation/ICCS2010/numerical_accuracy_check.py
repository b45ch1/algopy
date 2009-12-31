import numpy
import time
import algopy.utp.utpm as utpm

# D-1 is the degree of the polynomial
# P is the number of simultaneously computed decompositions (Single Program Multiple Data)
# M is the number of rows of the matrix A
# N is the number of columsn of the matrix A

D,P,M,N = 5,5,100,5

A = utpm.UTPM(numpy.random.rand(D,P,M,N))

# compute push forward of the QR decomposition
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

B = utpm.UTPM.dot(Q,R)
print 'largest error for all QR decompositions and derivative degrees is: ',numpy.max(A.data - B.data)


# check runtime
tic = time.time()
for p in range(P):
    A0 = numpy.ascontiguousarray(A.data[0,p])
    Q0,R0 = numpy.linalg.qr(A0)
toc = time.time()
runtime_normal = toc - tic

print 'measured runtime ratio push_forward/normal in sec: ', runtime_push_forward/runtime_normal

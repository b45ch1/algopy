import numpy
from algopy import CGraph, Function, UTPM, dot, qr, eigh, inv, solve

# first order derivatives, one directional derivative
# D - 1 is the degree of the Taylor polynomial
# P directional derivatives at once
# M number of rows of A
# N number of cols of A
D,P,M,N = 2,1,5,2

# generate badly conditioned matrix A
A = UTPM(numpy.zeros((D,P,M,N)))
x = UTPM(numpy.zeros((D,P,M,1)))
y = UTPM(numpy.zeros((D,P,M,1)))

x.data[0,0,:,0] = [1,1,1,1,1]
x.data[1,0,:,0] = [1,1,1,1,1]

y.data[0,0,:,0] = [1,2,1,2,1]
y.data[1,0,:,0] = [1,2,1,2,1]

alpha = 10**-5
A = dot(x,x.T) + alpha*dot(y,y.T)

A = A[:,:2]


# Method 1: Naive approach
Apinv = dot(inv(dot(A.T,A)),A.T)

print('naive approach: A Apinv A - A = 0 \n', dot(dot(A, Apinv),A) - A)
print('naive approach: Apinv A Apinv - Apinv = 0 \n', dot(dot(Apinv, A),Apinv) - Apinv)
print('naive approach: (Apinv A)^T - Apinv A = 0 \n', dot(Apinv, A).T  - dot(Apinv, A))
print('naive approach: (A Apinv)^T - A Apinv = 0 \n', dot(A, Apinv).T  - dot(A, Apinv))


# Method 2: Using the differentiated QR decomposition
Q,R = qr(A)
tmp1 = solve(R.T, A.T)
tmp2 = solve(R, tmp1)
Apinv = tmp2

print('QR approach: A Apinv A - A = 0 \n',  dot(dot(A, Apinv),A) - A)
print('QR approach: Apinv A Apinv - Apinv = 0 \n', dot(dot(Apinv, A),Apinv) - Apinv)
print('QR approach: (Apinv A)^T - Apinv A = 0 \n', dot(Apinv, A).T  - dot(Apinv, A))
print('QR approach: (A Apinv)^T - A Apinv = 0 \n', dot(A, Apinv).T  - dot(A, Apinv))

# Method 3: Stable evaluation of the analytical derivative formula

A0 = A.data[0,0]
A1 = A.data[1,0]

Q0, R0 = numpy.linalg.qr(A0)

# compute nominal solution
tmp1 = solve(R0.T, A0.T)
C0 = solve(R0, tmp1)

# compute first directional derivative
tmp2 = A1.T - dot( dot(A1.T, A0) + dot(A0.T, A1), C0)
tmp1 = solve(R0.T, tmp2)
C1 = solve(R0, tmp1)

Apinv.data[0,0] = C0
Apinv.data[1,0] = C1

print('analytical approach: A Apinv A - A = 0 \n',  dot(dot(A, Apinv),A) - A)
print('analytical approach: Apinv A Apinv - Apinv = 0 \n', dot(dot(Apinv, A),Apinv) - Apinv)
print('analytical approach: (Apinv A)^T - Apinv A = 0 \n', dot(Apinv, A).T  - dot(Apinv, A))
print('analytical approach: (A Apinv)^T - A Apinv = 0 \n', dot(A, Apinv).T  - dot(A, Apinv))

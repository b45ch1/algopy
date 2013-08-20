import numpy; from algopy import UTPM

# symmetric eigenvalue decomposition, forward UTPM
D,P,M,N = 3,1,4,4
Q,R = UTPM.qr(UTPM(numpy.random.rand(D,P,M,N)))
l = UTPM(numpy.random.rand(*(D,P,N)))
l.data[0,0,:4] = [1,1,2,3]
l.data[1,0,:4] = [0,0,3,4]
l.data[2,0,:4] = [1,2,5,6]
L = UTPM.diag(l)
B = UTPM.dot(Q,UTPM.dot(L,Q.T))

print('B = \n', B)
l2,Q2 = UTPM.eigh(B)
print('l2 - l =\n',l2 - l)

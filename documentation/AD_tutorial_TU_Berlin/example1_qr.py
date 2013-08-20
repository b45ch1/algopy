import numpy; from algopy import UTPM

# QR decomposition, UTPM forward
D,P,M,N = 3,1,500,20
A = UTPM(numpy.random.rand(D,P,M,N))
Q,R = UTPM.qr(A)
B = UTPM.dot(Q,R)

# check that the results are correct
print('Q.T Q - 1\n',UTPM.dot(Q.T,Q) - numpy.eye(N))
print('QR - A\n',B - A)
print('triu(R) - R\n', UTPM.triu(R) - R)

# QR decomposition, UTPM reverse
Bbar = UTPM(numpy.random.rand(D,P,M,N))
Qbar,Rbar = UTPM.pb_dot(Bbar, Q, R, B)
Abar = UTPM.pb_qr(Qbar, Rbar, A, Q, R)

print('Abar - Bbar\n',Abar - Bbar)

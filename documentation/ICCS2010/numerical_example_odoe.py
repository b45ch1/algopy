import numpy
from algopy.utp.utpm import *

# Np = number of parameters p
# Nq = number of control variables q
# Nm = number of measurements
# P  = number of vectorized operations at once (SPMD)

D,Nq,Np,Nm = 2,3,3,10
P = Np
q = UTPM(numpy.random.rand(D,P,Nq))
p = UTPM(numpy.random.rand(D,P,Np))

# STEP 1: compute push forward
F = UTPM(numpy.zeros((D,P,Nm)))
for nm in range(Nm):
    F[nm] =  numpy.sum([ numpy.random.rand() * q[n]*p[-n] for n in range(Nq)])

J = F.FtoJT().T
Q,R = UTPM.qr(J)
Id = numpy.eye(P)
RT = R.T
D = UTPM.solve(RT,Id)
C = UTPM.solve(D,R)
l,U = UTPM.eig(C)

arg = UTPM.argmax(l)

# STEP 2: compute pullback
lbar = UTPM(numpy.zeros(l.data.shape))
lbar.data[0,0, arg] = 1.
Ubar = UTPM(numpy.zeros(U.data.shape))

Cbar = UTPM.eig_pullback(lbar, Ubar, C, l, U)
Dbar, Rbar = UTPM.solve_pullback( Cbar, D, R, C)

RTbar, Idbar = UTPM.solve_pullback( Dbar, RT, Id, D)
Rbar = RTbar.T

Qbar = UTPM(numpy.zeros(Q.data.shape))
Jbar = UTPM.qr_pullback(Qbar, Rbar, J, Q, R)

print Jbar
 

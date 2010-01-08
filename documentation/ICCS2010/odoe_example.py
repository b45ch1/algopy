import numpy
from numpy.testing import *
from prettyplotting import *
from algopy.utp.utpm import *

# Np = number of parameters p
# Nq = number of control variables q
# Nm = number of measurements
# P  = number of vectorized operations at once (SPMD)

D,Nq,Np,Nm = 2,1,2,10
P = Np
q = UTPM(numpy.zeros((D,P,Nq)))
q0 = 1.
q.data[0,:,0] = q0
p = UTPM(numpy.zeros((D,P,Np)))
p0 = numpy.random.rand(Np)
p.data[0,0,:] = p0
p.data[0,1,:] = p0
p.data[1,:,:] = numpy.eye(P)

B = UTPM(numpy.zeros((D,P,Nm, Np)))
B0 = numpy.random.rand(Nm,Np)
B.data[0,0] = B0 
B.data[0,1] = B0

# STEP 1: compute push forward
G = UTPM.dot(B, p)
F  = G * q[0]

J = F.FtoJT().T
assert_array_almost_equal( J.data[0,0], B0)

# Q,R = UTPM.qr(J)
# Id = numpy.eye(P)
# RT = R.T
# D = UTPM.solve(RT,Id)
# C = UTPM.solve(D,R)
E = UTPM.dot(J.T,J)
C = UTPM.inv(E)

assert_array_almost_equal( C.data[0,0], q0**-2 * numpy.linalg.inv(numpy.dot(B0.T,B0)))

l,U = UTPM.eigh(C)
l0, U0 = numpy.linalg.eigh(q0**-2 * numpy.linalg.inv(numpy.dot(B0.T,B0)))

assert_array_almost_equal(l.data[0,0], l0)
assert_array_almost_equal(U.data[0,0], U0)

arg = UTPM.argmax(l)

# STEP 2: compute pullback
lbar = UTPM(numpy.zeros(l.data.shape))
lbar.data[0,0, arg] = 1.
Ubar = UTPM(numpy.zeros(U.data.shape))

Cbar = UTPM.eigh_pullback(lbar, Ubar, C, l, U)

# # Dbar, Rbar = UTPM.solve_pullback( Cbar, D, R, C)
# # RTbar, Idbar = UTPM.solve_pullback( Dbar, RT, Id, D)
# # Rbar = RTbar.T
# # Qbar = UTPM(numpy.zeros(Q.data.shape))
# # Jbar = UTPM.qr_pullback(Qbar, Rbar, J, Q, R)

# print Cbar

Ebar = UTPM.inv_pullback(Cbar, C, E)
print Ebar
# JTbar, Jbar = UTPM.dot_pullback(Ebar, J.T, J, E)
# print JTbar, Jbar
# Fbar = Jbar.T.JTtoF()
# qbar = UTPM.dot(Fbar.T, G)

# print qbar



# #############################################
# # analytical solution
# #############################################
# c = numpy.max(numpy.linalg.eig( numpy.linalg.inv(numpy.dot(B0.T, B0)))[0])
# dPhidq = - 2* c * q.data[0,0,0]**-3

# # print dPhidq











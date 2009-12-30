from algopy.utp.utpm import *

# Np = number of parameters p
# Nq = number of control variables q
# Nm = number of measurements
# P  = number of vectorized operations at once (SPMD)

D,Nq,Np,Nm = 2,3,3,10
P = Np
q = UTPM(numpy.random.rand(D,P,Nq))
p = UTPM(numpy.random.rand(D,P,Np))

F = UTPM(numpy.zeros((D,P,Nm)))

for nm in range(Nm):
    F[nm] =  numpy.sum([ (nm+1.)*n * q[n]*p[-n] for n in range(Nq)])

J = F.FtoJT().T
Q,R = UTPM.qr(J)
Id = numpy.eye(P)
D = UTPM.solve(R.T,Id)
C = UTPM.solve(D,R)
l,U = UTPM.eig(C)

print l
l11 = l.max()

print l11
 

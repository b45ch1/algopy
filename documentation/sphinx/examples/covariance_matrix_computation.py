import numpy
from algopy import CGraph, Function, UTPM, dot, qr, qr_full, eigh, inv, solve, zeros

def eval_covariance_matrix_naive(J1, J2):
    M,N = J1.shape
    K,N = J2.shape
    tmp = zeros((N+K, N+K), dtype=J1)
    tmp[:N,:N] = dot(J1.T,J1)
    tmp[:N,N:] = J2.T
    tmp[N:,:N] = J2
    return inv(tmp)[:N,:N]
    
def eval_covariance_matrix_qr(J1, J2):
    M,N = J1.shape
    K,N = J2.shape
    Q,R = qr_full(J2.T)
    Q2 = Q[:,K:].T
    J1_tilde = dot(J1,Q2.T)
    Q,R = qr(J1_tilde)
    V = solve(R.T, Q2)
    return dot(V.T,V)


# dimensions of the involved matrices
D,P,M,N,K,Nx = 2,1,5,3,1,1

# trace the function evaluation of METHOD 1: nullspace method
cg1 = CGraph()
J1 = Function(UTPM(numpy.random.rand(*(D,P,M,N))))
J2 = Function(UTPM(numpy.random.rand(*(D,P,K,N))))
C = eval_covariance_matrix_qr(J1, J2)
y = C[0,0]
cg1.trace_off()
cg1.independentFunctionList = [J1, J2]
cg1.dependentFunctionList = [y]
print('covariance matrix: C =\n',C)

# trace the function evaluation of METHOD 2: naive method (potentially numerically unstable)
cg2 = CGraph()
J1 = Function(J1.x)
J2 = Function(J2.x)
C2 = eval_covariance_matrix_naive(J1, J2)
y = C2[0,0]
cg2.trace_off()
cg2.independentFunctionList = [J1, J2]
cg2.dependentFunctionList = [y]
print('covariance matrix: C =\n',C2)

# check that both algorithms returns the same result
print('difference between naive and nullspace method:\n',C - C2)

# compute the gradient for another value of J1 and J2
J1 = numpy.random.rand(*(M,N))
J2 = numpy.random.rand(*(K,N))

g1 = cg1.gradient([J1,J2])
g2 = cg2.gradient([J1,J2])

print('naive approach: dy/dJ1 = ', g1[0])
print('naive approach: dy/dJ2 = ', g1[1])

print('nullspace approach: dy/dJ1 = ', g2[0])
print('nullspace approach: dy/dJ2 = ', g2[1])






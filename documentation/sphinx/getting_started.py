from numpy import array, zeros, eye, exp
from algopy import UTPM

def f(x):
    """ some function """
    return x[0]*x[1]*x[2] + exp(x[0])*x[1]
    
def g(x):
    """ symbolically derived gradient function of f(x) """
    return array([ x[1]*x[2] + exp(x[0])*x[1],
                   x[0]*x[2] + exp(x[0]),
                   x[0]*x[1]])

D,P,N = 2,3,3
x = UTPM(zeros((D,P,N)))
x.data[0,:] = [3,5,7]
x.data[1,:,:] = eye(P)
ALGOPY_gradient = f(x).data[1]
symbolic_gradient = g([3,5,7])
print 'gradient computed with ALGOPY using UTP arithmetic = ',ALGOPY_gradient
print 'evaluated symbolic gradient = ',symbolic_gradient
print 'difference =', ALGOPY_gradient - symbolic_gradient




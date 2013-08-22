"""
compute the Hessian H of the function
def f(x):
    return sin(x[0] + cos(x[1])*x[0])
at x = (3,7)
"""

import numpy; from numpy import sin,cos, array, zeros
from taylorpoly import UTPS
def f_fcn(x):
    return sin(x[0] + cos(x[1])*x[0])

S = array([[1,0,1],[0,1,1]], dtype=float)
P = S.shape[1]
print('seed matrix with P = %d directions S = \n'%P, S)
x1 = UTPS(zeros(1+2*P), P = P)
x2 = UTPS(zeros(1+2*P), P = P)
x1.data[0] = 3; x1.data[1::2] = S[0,:]
x2.data[0] = 7; x2.data[1::2] = S[1,:]
y = f_fcn([x1,x2])
print('x1=',x1);  print('x2=',x2); print('y=',y)
H = zeros((2,2),dtype=float)
H[0,0] = 2*y.coeff[0,2]
H[1,0] = H[0,1] = (y.coeff[2,2] - y.coeff[0,2] - y.coeff[1,2])
H[1,1] =  2*y.coeff[1,2]

def H_fcn(x):
    H11 = -(1+cos(x[1]))**2*sin(x[0]+cos(x[1])*x[0])
    H21 = -sin(x[1]) * cos(x[0] + cos(x[1])*x[0]) \
          +sin(x[1]) *x[0]*(1+ cos(x[1]))*sin(x[0]+cos(x[1])*x[0])
    H22 = -cos(x[1])*x[0]*cos(x[0]+cos(x[1])*x[0])\
          -(sin(x[1])*x[0])**2*sin(x[0]+cos(x[1])*x[0])
    return array([[H11, H21],[H21,H22]])

print('symbolic Hessian - AD Hessian = \n', H - H_fcn([3,7]))
print('exact interpolation Hessian H(x_0) = \n', H)







import numpy; from numpy import sin,cos; from taylorpoly import UTPS

def f(x):
    return sin(x[0] + cos(x[1])*x[0]) + x[1]*x[0]

x = [UTPS([3,1,0],P=2), UTPS([7,0,1],P=2)]
y = f(x)

print('normal function evaluation y_0 = f(x_0) = ', y.data[0])
print('gradient evaluation g(x_0) = ', y.data[1:])






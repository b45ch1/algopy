import numpy
import algopy

def f_fcn(x):
    A = algopy.zeros((2,2), dtype=x)
    A[0,0] = x[0]
    A[1,0] = x[1] * x[0]
    A[0,1] = x[1]
    Q,R = algopy.qr(A)
    return R[0,0]


# Method 1: Complex-step derivative approximation (CSDA)
h = 10**-20
x0 = numpy.array([3,2],dtype=float)
x1 = numpy.array([1,0])
yc = numpy.imag(f_fcn(x0 + 1j * h * x1) - f_fcn(x0 - 1j * h * x1))/(2*h)

# Method 2: univariate Taylor polynomial arithmetic (UTP)
ax = algopy.UTPM(numpy.zeros((2,1,2)))
ax.data[0,0] = x0
ax.data[1,0] = x1
ay = f_fcn(ax)

# Method 3: finite differences
h = 10**-8
yf = (f_fcn(x0 + h * x1) - f_fcn(x0))/h

# Print results
print('CSDA result =',yc)
print('UTP result  =',ay.data[1,0])
print('FD  result  =',yf)


import numpy
from algopy import UTPM, dot

D,P,M,N = 2,1,3,4
x_data = 7*numpy.arange(D*P*M*N,dtype=float).reshape((D,P,M,N))
y_data = numpy.arange(D*P*M*N,dtype=float).reshape((D,P,N,M))

# calling algorithms directly
z_data = numpy.zeros((D,P,M,M))
UTPM._dot(x_data, y_data, z_data)

# use UTPM instance
x = UTPM(x_data)
y = UTPM(y_data)
z = dot(x, y)

print('z.data - z_data', z.data - z_data)

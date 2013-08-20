import numpy
from algopy import CGraph, Function, UTPM, dot, qr, eigh, inv, zeros

def f(y):
    retval = zeros((3,1),dtype=y)
    retval[0,0] = numpy.log(dot(y.T,y))
    retval[1,0] = numpy.exp(dot(y.T,y))
    retval[2,0] = numpy.exp(dot(y.T,y)) -  numpy.log(dot(y.T,y))
    return retval
    
D,Nm = 2,40
P = Nm
y = UTPM(numpy.zeros((2,P,Nm)))

y.data[0,:] = numpy.random.rand(Nm)
y.data[1,:] = numpy.eye(Nm)


# print f(y)
J = f(y).data[1,:,:,0]
print('Jacobian J(y) = \n', J)

C_epsilon = 0.3*numpy.eye(Nm)

print(J.shape)

C = dot(J.T, dot(C_epsilon,J))

print('Covariance matrix of z: C = \n',C)
        
    
        

    
 

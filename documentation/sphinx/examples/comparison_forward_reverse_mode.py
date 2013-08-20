import numpy
from algopy import CGraph, Function, UTPM, dot, qr, eigh, inv, zeros

def f(x,y):
    return dot(x.T, y) +  dot((x*y-x).T, (x-y))


# create an UTPM instance
D,N,M = 2,3,2
P = 2*N

x = UTPM(numpy.zeros((D,P,2*N,1)))
x.data[0,:] = numpy.random.rand(2*N,1)
x.data[1,:,:,0] = numpy.eye(P)
y = x[N:]
x = x[:N]

# wrap the UTPM instance in a Function instance to trace all operations 
# that have x as an argument
# create a CGraph instance that to store the computational trace
cg = CGraph().trace_on()
x = Function(x)
y = Function(y)
z = f(x,y)
cg.trace_off()

# define dependent and independent variables in the computational procedure
cg.independentFunctionList = [x,y]
cg.dependentFunctionList = [z]

# Since the UTPM instrance is wrapped in a Function instance we have to access it
# by y.x. That means the Jacobian is
grad1 = z.x.data[1,:,0]

print('forward gradient g(x) = \n', grad1)

# Now we want to compute the same Jacobian in the reverse mode of AD
# before we do that we have a look what the computational graph looks like:
# print 'Computational graph is', cg

# the reverse mode is called by cg.pullback([ybar])
# it is a little hard to explain what's going on here. Suffice to say that we
# now compute one row of the Jacobian instead of one column as in the forward mode

zbar = z.x.zeros_like()

# compute gradient in the reverse mode
zbar.data[0,:,0,0] = 1
cg.pullback([zbar])
grad2_x = x.xbar.data[0,0]
grad2_y = y.xbar.data[0,0]
grad2 = numpy.concatenate([grad2_x, grad2_y])

print('reverse gradient g(x) = \n', grad2)

#check that the forward computed gradient equals the reverse gradient
print('difference forward/reverse gradient=\n',grad1 - grad2)

# one can also easiliy extract the Hessian
H = numpy.zeros((2*N,2*N))
H[:,:N] = x.xbar.data[1,:,:,0]
H[:,N:] = y.xbar.data[1,:,:,0]

print('Hessian = \n', H)










import algopy, numpy
import algopy.exact_interpolation as ei

def eval_F(x):
    retval = algopy.zeros(3, dtype=x)
    retval[0] = x[0]*x[1]
    retval[1] = x[1]*x[2]
    retval[2] = x[0] - x[1]
    return retval

D,M,N = 3,3,3
P = 2*M*N

# STEP 1: generate all necessary directions
Id = numpy.eye(N)
dirs = numpy.zeros((P, N))

count = 0
for n1 in range(N):
    for n2 in range(N): 
        dirs[count] += Id[n1]
        dirs[count] += Id[n2]
        
        dirs[count + 1] += Id[n1]
        dirs[count + 1] -= Id[n2]
        
        count += 2

print('dirs =')
print(dirs)

# STEP 2: use these directions to initialize the UTPM instance
xdata = numpy.zeros((D,2*N*N,N))
xdata[0] = [1,2,3]
xdata[1] = dirs

# STEP 3: compute function F in UTP arithmetic
x = algopy.UTPM(xdata)
y = eval_F(x)

# STEP 4: use polarization identity to build univariate Taylor polynomial
#         of the Jacobian J
Jdata = numpy.zeros((D-1, N, N, N))
count, count2 = 0,0

# build J_0
for n in range(N):
    Jdata[0,:,:,n] = y.data[1, 2*n*(N+1), :]/2.
    
# build J_1
count = 0
for n1 in range(N):
    for n2 in range(N):
        Jdata[1,n2,:,n1] = 0.5*( y.data[2, count] - y.data[2, count+1])
        count += 2

# initialize UTPM instance
J = algopy.UTPM(Jdata)

# STEP 5: evaluate Phi in UTP arithmetic
Phi = algopy.trace(algopy.dot(J.T, J))
print('Phi=',Phi)
print('gradient of Phi =', Phi.data[1,:])



from numpy.testing import *
import numpy
from algopy import *
import adolc
import time
reps = 100
N,P,D = 100,1,2
def f(x):
    return numpy.sum([x[i] for i in range(N)])
    
# trace with ALGOPY
start_time = time.time()
cg = CGraph()
x = Function(UTPM(numpy.random.rand(1,1,N)))
y = f(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]
end_time = time.time()
time_trace_algopy =  end_time - start_time

# trace with PYADOLC
start_time = time.time()
adolc.trace_on(1)
x = adolc.adouble(numpy.random.rand(N))
adolc.independent(x)
y = f(x)
adolc.dependent(y)
adolc.trace_off()
end_time = time.time()
time_trace_adolc = end_time - start_time

# trace with PYADOLC.cgraph
from adolc.cgraph import AdolcProgram
start_time = time.time()
ap = AdolcProgram()
ap.trace_on(2)
x = adolc.adouble(numpy.random.rand(N))
ap.independent(x)
y = f(x)
ap.dependent(y)
ap.trace_off()
end_time = time.time()
time_trace_cgraph = end_time - start_time

# time ALGOPY hos_forward
x = UTP(numpy.random.rand(D,N))
start_time = time.time()
for rep in range(reps):
    cg.pushforward([x])
end_time = time.time()
time_hos_forward_algopy = end_time - start_time

# time PYADOLC hos_forward
x = numpy.random.rand(N)
V = numpy.random.rand(N,D-1)

start_time = time.time()
for rep in range(reps):
    adolc.hos_forward(1,x , V, keep=0)
end_time = time.time()
time_hos_forward_adolc = end_time - start_time

# time PYADOLC.cgraph hos_forward
x = numpy.random.rand(N)
V = numpy.random.rand(N,1,D-1)
for rep in range(reps):
    ap.forward([x],[V])
end_time = time.time()
time_hos_forward_cgraph = end_time - start_time

# time ALGOPY hov_forward
x = UTPM(numpy.random.rand(D,P,N))
start_time = time.time()
for rep in range(reps):
    cg.pushforward([x])
end_time = time.time()
time_hov_forward_algopy = end_time - start_time

# time PYADOLC hov_forward
x = numpy.random.rand(N)
V = numpy.random.rand(N,P,D-1)

start_time = time.time()
for rep in range(reps):
    adolc.hov_forward(1,x, V)
end_time = time.time()
time_hov_forward_adolc = end_time - start_time

# time PYADOLC.cgraph hos_forward
x = numpy.random.rand(N)
V = numpy.random.rand(N,P,D-1)
for rep in range(reps):
    ap.forward([x],[V])
end_time = time.time()
time_hov_forward_cgraph = end_time - start_time

# time ALGOPY hov_reverse
ybar = UTPM(numpy.random.rand(D,P))
start_time = time.time()
for rep in range(reps):
    cg.pullback([ybar])
end_time = time.time()
time_hov_reverse_algopy = end_time - start_time

# time PYADOLC hov_reverse
W = numpy.random.rand(1,1,D)
start_time = time.time()
V = numpy.random.rand(N,P,D-1)
for rep in range(reps):
    for p in range(P):
        adolc.hos_forward(1,x, V[:,p,:], keep=D)
        adolc.hov_ti_reverse(1, W)
end_time = time.time()
time_hov_reverse_adolc = end_time - start_time

# time PYADOLC.cgraph hov_reverse
W = numpy.random.rand(1, 1,P,D)
for rep in range(reps):
    ap.reverse([W])
end_time = time.time()
time_hov_reverse_cgraph = end_time - start_time

print('----------------')
print(time_trace_algopy)
print(time_trace_adolc)
print(time_trace_cgraph)
print('----------------')
print(time_hos_forward_algopy)
print(time_hos_forward_adolc)
print(time_hos_forward_cgraph)
print('----------------')
print(time_hov_forward_algopy)
print(time_hov_forward_adolc)
print(time_hov_forward_cgraph)
print('----------------')
print(time_hov_reverse_algopy)
print(time_hov_reverse_adolc)
print(time_hov_reverse_cgraph) 

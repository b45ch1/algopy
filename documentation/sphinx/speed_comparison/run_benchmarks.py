import numpy as np
import time
N = 30

import benchmark1
import use_adolc
import use_scientific
import use_uncertainties
import use_algopy
import use_numdifftools

# pyadolc
f = benchmark1.F(N)
a = use_adolc.EVAL(f, np.ones(N))

start_time = time.time()
ref_g =  a.gradient(3*np.ones(N))
end_time = time.time()
print 'runtime = ',end_time - start_time

start_time = time.time()
ref_H =  a.hessian(3*np.ones(N))
end_time = time.time()
print 'runtime = ',end_time - start_time


# scientifc
f = benchmark1.F(N)
a = use_scientific.EVAL(f, np.ones(N))

start_time = time.time()
g = a.gradient(3*np.ones(N))
end_time = time.time()
print 'runtime = ',end_time - start_time
print 'correct = ', np.linalg.norm(g - ref_g)


# algopy, UTPS variant
f = benchmark1.F(N)
a = use_algopy.EVAL(f, np.ones(N))

start_time = time.time()
g = a.gradient(3*np.ones(N))
end_time = time.time()
print 'runtime = ',end_time - start_time
print 'correct = ', np.linalg.norm(g - ref_g)


# algopy, UTPM variant
f = benchmark1.G(N)
a = use_algopy.EVAL2(f, np.ones(N))

start_time = time.time()
g = a.gradient(3*np.ones(N))
end_time = time.time()
print 'runtime = ',end_time - start_time
print 'correct = ', np.linalg.norm(g - ref_g)


# algopy, forward UTPM variant
f = benchmark1.G(N)
a = use_algopy.EVAL2(f, np.ones(N))

start_time = time.time()
g = a.forwardgradient(3*np.ones(N))
end_time = time.time()
print 'runtime = ',end_time - start_time
print 'correct = ', np.linalg.norm(g - ref_g)


# numdifftools
f = benchmark1.F(N)
a = use_numdifftools.EVAL(f, np.ones(N))

start_time = time.time()
g =  a.gradient(3*np.ones(N))
end_time = time.time()
print 'runtime = ',end_time - start_time
print 'correct = ', np.linalg.norm(g - ref_g)


start_time = time.time()
H =  a.hessian(3*np.ones(N))
end_time = time.time()
print 'runtime = ',end_time - start_time
print 'correct = ', np.linalg.norm(H - ref_H)

# uncertainties
f = benchmark1.F(N)
a = use_uncertainties.EVAL(f, np.ones(N))

start_time = time.time()
g = a.gradient(3*np.ones(N))
end_time = time.time()
print 'runtime = ',end_time - start_time
print 'correct = ', np.linalg.norm(g - ref_g)



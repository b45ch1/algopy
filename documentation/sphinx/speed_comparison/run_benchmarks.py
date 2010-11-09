import numpy as np
import time

import benchmark1
import use_adolc
import use_scientific
import use_uncertainties
import use_algopy
import use_numdifftools




method = {'algopy_reverse_utps':0, 'algopy_reverse_utpm':1, 'pyadolc':2, 'scientific':3, 'uncertainties':4, 'numdifftools':5, 'algopy_forward_utpm':6,}

# GRADIENT COMPUTATION
# --------------------

gradient_N_list = [1,2,4,8,16,32,64,128]
# gradient_N_list = [1,2]

results_gradient_list = []
for N in gradient_N_list:
    print 'N=',N
    results_gradient = np.zeros((7,2))
    
    # pyadolc
    f = benchmark1.F(N)
    a = use_adolc.EVAL(f, np.ones(N))
    start_time = time.time()
    ref_g =  a.gradient(3*np.ones(N))
    end_time = time.time()
    results_gradient[method['pyadolc']] = end_time - start_time, 0
    
    # scientifc
    f = benchmark1.F(N)
    a = use_scientific.EVAL(f, np.ones(N))
    start_time = time.time()
    g = a.gradient(3*np.ones(N))
    end_time = time.time()
    results_gradient[method['scientific']] = end_time - start_time, np.linalg.norm(g - ref_g)
    
    # algopy, UTPS variant
    f = benchmark1.F(N)
    a = use_algopy.EVAL(f, np.ones(N))
    start_time = time.time()
    g = a.gradient(3*np.ones(N))
    end_time = time.time()
    results_gradient[method['algopy_reverse_utps']] = end_time - start_time, np.linalg.norm(g - ref_g)
    
    # algopy, UTPM variant
    f = benchmark1.G(N)
    a = use_algopy.EVAL2(f, np.ones(N))
    start_time = time.time()
    g = a.gradient(3*np.ones(N))
    end_time = time.time()
    results_gradient[method['algopy_reverse_utpm']] = end_time - start_time, np.linalg.norm(g - ref_g)
    
    # algopy, forward UTPM variant
    f = benchmark1.G(N)
    a = use_algopy.EVAL2(f, np.ones(N))
    start_time = time.time()
    g = a.forwardgradient(3*np.ones(N))
    end_time = time.time()
    results_gradient[method['algopy_forward_utpm']] = end_time - start_time, np.linalg.norm(g - ref_g)
    
    # numdifftools
    f = benchmark1.F(N)
    a = use_numdifftools.EVAL(f, np.ones(N))
    start_time = time.time()
    g =  a.gradient(3*np.ones(N))
    end_time = time.time()
    results_gradient[method['numdifftools']] = end_time - start_time, np.linalg.norm(g - ref_g)
    
    # uncertainties
    f = benchmark1.F(N)
    a = use_uncertainties.EVAL(f, np.ones(N))
    start_time = time.time()
    g = a.gradient(3*np.ones(N))
    end_time = time.time()
    results_gradient[method['uncertainties']] = end_time - start_time, np.linalg.norm(g - ref_g)
    
    results_gradient_list.append(results_gradient)

results_gradients = np.array(results_gradient_list)


# HESSIAN COMPUTATION
# -------------------


results_hessian_list = []
hessian_N_list = [1,2,4,8,16,32,64]
# hessian_N_list = [1,2]

for N in hessian_N_list:
    print 'N=',N
    results_hessian = np.zeros((7,2))
    
    # pyadolc
    f = benchmark1.F(N)
    a = use_adolc.EVAL(f, np.ones(N))
    start_time = time.time()
    ref_H =  a.hessian(3*np.ones(N))
    end_time = time.time()
    results_hessian[method['pyadolc']] = end_time - start_time, 0
    
    # numdifftools
    f = benchmark1.F(N)
    a = use_numdifftools.EVAL(f, np.ones(N))
    start_time = time.time()
    H =  a.hessian(3*np.ones(N))
    end_time = time.time()
    results_hessian[method['numdifftools']] = end_time - start_time, 0
    results_hessian_list.append(results_hessian)

results_hessians = np.array(results_hessian_list)



# PLOT RESULTS

print results_gradients.shape

import matplotlib.pyplot as pyplot
import prettyplotting

# plot gradient run times
pyplot.figure()
pyplot.title('Gradient run times')
pyplot.semilogy(gradient_N_list, results_gradients[:,method['pyadolc'],0], '-ko', markerfacecolor='None', label = 'pyadolc')
pyplot.semilogy(gradient_N_list, results_gradients[:,method['algopy_reverse_utps'],0], '-.k<', markerfacecolor='None', label = 'algopy reverse utps')
pyplot.semilogy(gradient_N_list, results_gradients[:,method['algopy_reverse_utpm'],0], '-.k>', markerfacecolor='None', label = 'algopy reverse utpm')
pyplot.semilogy(gradient_N_list, results_gradients[:,method['algopy_forward_utpm'],0], '-.kv', markerfacecolor='None', label = 'algopy forward utpm')
pyplot.semilogy(gradient_N_list, results_gradients[:,method['uncertainties'],0], '--kh', markerfacecolor='None', label = 'uncertainties')
pyplot.semilogy(gradient_N_list, results_gradients[:,method['numdifftools'],0], '--ks', markerfacecolor='None', label = 'numdifftools')
pyplot.ylabel('time $t$')
pyplot.xlabel('problem size $N$')
pyplot.grid()
leg = pyplot.legend(loc='best')
frame= leg.get_frame()
frame.set_alpha(0.4)
pyplot.savefig('gradient_runtimes.png')


# plot hessian run times
pyplot.figure()
pyplot.title('Hessian run times')
pyplot.semilogy(hessian_N_list, results_hessians[:,method['pyadolc'],0], '-ko', markerfacecolor='None', label = 'pyadolc')
pyplot.semilogy(hessian_N_list, results_hessians[:,method['numdifftools'],0], '--ks', markerfacecolor='None', label = 'numdifftools')
pyplot.ylabel('time $t$')
pyplot.xlabel('problem size $N$')
pyplot.grid()
leg = pyplot.legend(loc='best')
frame= leg.get_frame()
frame.set_alpha(0.4)
pyplot.savefig('hessian_runtimes.png')

# pyplot.show()



import numpy as np
import time

import benchmark1
import use_adolc
import use_scientific
import use_uncertainties
import use_algopy
import use_numdifftools
import use_funcdesigner




method = {'algopy_reverse_utps':0, 'algopy_reverse_utpm':1, 'pyadolc':2, 'scientific':3, 'uncertainties':4, 'numdifftools':5, 'algopy_forward_utpm':6, 'python':7, 'algopy_forward_utps':8, 'funcdesigner':9}


# FUNCTION COMPUTATION
# --------------------

function_N_list = [1,2,4,8,16,32,64,128,256]
function_N_list = [1,2]


results_function_list = []
for N in function_N_list:
    print 'N=',N
    results_function = np.zeros((10,2))
    
    # pure python
    f = benchmark1.F(N)
    start_time = time.time()
    ref_f =  f(3*np.ones(N))
    end_time = time.time()
    results_function[method['python']] = end_time - start_time, 0
    # print ref_f
    
    # pyadolc
    f = benchmark1.F(N)
    a = use_adolc.EVAL(f, np.ones(N))
    start_time = time.time()
    f =  a.function(3*np.ones(N))
    end_time = time.time()
    results_function[method['pyadolc']] = end_time - start_time,  0
    # print f
    
    # algopy utps version
    f = benchmark1.F(N)
    a = use_algopy.EVAL(f, np.ones(N))
    start_time = time.time()
    f =  a.function(3*np.ones(N))
    end_time = time.time()
    results_function[method['algopy_forward_utps']] = end_time - start_time, 0
    
    # algopy utps version
    f = benchmark1.G(N)
    a = use_algopy.EVAL2(f, np.ones(N))
    start_time = time.time()
    f =  a.function(3*np.ones(N))
    end_time = time.time()
    results_function[method['algopy_forward_utpm']] = end_time - start_time, 0
    
    # funcdesigner
    f = benchmark1.F(N)
    a = use_funcdesigner.EVAL(f, np.ones(N))
    start_time = time.time()
    f =  a.function(3*np.ones(N))
    end_time = time.time()
    results_function[method['funcdesigner']] = end_time - start_time,  0
    
    results_function_list.append(results_function)
    # print f

results_functions = np.array(results_function_list)


# GRADIENT COMPUTATION
# --------------------

gradient_N_list = [2,4,8,16,32,64,96,128]
# gradient_N_list = [2,4]

results_gradient_list = []
for N in gradient_N_list:
    print 'N=',N
    results_gradient = np.zeros((10,2))
    
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
    
    # funcdesigner
    f = benchmark1.F(N)
    a = use_funcdesigner.EVAL(f, np.ones(N))
    start_time = time.time()
    g = a.gradient(3*np.ones(N))
    end_time = time.time()
    results_gradient[method['funcdesigner']] = end_time - start_time, np.linalg.norm(g - ref_g)
    
    
    
    results_gradient_list.append(results_gradient)

results_gradients = np.array(results_gradient_list)


# HESSIAN COMPUTATION
# -------------------

results_hessian_list = []
hessian_N_list = [1,2,4,8,16,32,64,96]
hessian_N_list = [1,2]

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
    
    # algopy forward utpm variant
    f = benchmark1.G(N)
    a = use_algopy.EVAL2(f, np.ones(N))
    start_time = time.time()
    g = a.forwardhessian(3*np.ones(N))
    end_time = time.time()
    results_hessian[method['algopy_forward_utpm']] = end_time - start_time, 0
    
    # algopy forward/reverse utpm variant
    f = benchmark1.G(N)
    a = use_algopy.EVAL2(f, np.ones(N))
    start_time = time.time()
    g = a.hessian(3*np.ones(N))
    end_time = time.time()
    results_hessian[method['algopy_reverse_utpm']] = end_time - start_time, 0

results_hessians = np.array(results_hessian_list)



# PLOT RESULTS

print results_gradients.shape

import matplotlib.pyplot as pyplot
import prettyplotting

# plot gradient run times
pyplot.figure()
pyplot.title('Function run times')
pyplot.loglog(function_N_list, results_functions[:,method['pyadolc'],0], '-ko', markerfacecolor='None', label = 'pyadolc')
pyplot.loglog(function_N_list, results_functions[:,method['python'],0], '-.k<', markerfacecolor='None', label = 'python')
pyplot.loglog(function_N_list, results_functions[:,method['algopy_forward_utps'],0], '-.k<', markerfacecolor='None', label = 'algopy scalar')
pyplot.loglog(function_N_list, results_functions[:,method['algopy_forward_utpm'],0], '-.k>', markerfacecolor='None', label = 'algopy matrix')
pyplot.loglog(function_N_list, results_functions[:,method['funcdesigner'],0], '-.ks', markerfacecolor='None', label = 'funcdesigner')



pyplot.ylabel('time $t$ [seconds]')
pyplot.xlabel('problem size $N$')
pyplot.grid()
leg = pyplot.legend(loc=2)
frame= leg.get_frame()
frame.set_alpha(0.4)
pyplot.savefig('function_runtimes.png')

# plot gradient run times
pyplot.figure()
pyplot.title('Gradient run times')
pyplot.loglog(gradient_N_list, results_gradients[:,method['pyadolc'],0], '-ko', markerfacecolor='None', label = 'pyadolc')
pyplot.loglog(gradient_N_list, results_gradients[:,method['algopy_reverse_utps'],0], '-.k<', markerfacecolor='None', label = 'algopy reverse utps')
pyplot.loglog(gradient_N_list, results_gradients[:,method['algopy_reverse_utpm'],0], '-.k>', markerfacecolor='None', label = 'algopy reverse utpm')
pyplot.loglog(gradient_N_list, results_gradients[:,method['algopy_forward_utpm'],0], '-.kv', markerfacecolor='None', label = 'algopy forward utpm')
pyplot.loglog(gradient_N_list, results_gradients[:,method['uncertainties'],0], '--kh', markerfacecolor='None', label = 'uncertainties')
pyplot.loglog(gradient_N_list, results_gradients[:,method['numdifftools'],0], '--ks', markerfacecolor='None', label = 'numdifftools')
pyplot.loglog(gradient_N_list, results_gradients[:,method['funcdesigner'],0], '--kd', markerfacecolor='None', label = 'funcdesigner')

pyplot.ylabel('time $t$ [seconds]')
pyplot.xlabel('problem size $N$')
pyplot.grid()
leg = pyplot.legend(loc=2)
frame= leg.get_frame()
frame.set_alpha(0.4)
pyplot.savefig('gradient_runtimes.png')


# plot hessian run times
pyplot.figure()
pyplot.title('Hessian run times')
pyplot.loglog(hessian_N_list, results_hessians[:,method['pyadolc'],0], '-ko', markerfacecolor='None', label = 'pyadolc (fo/rev)')
pyplot.loglog(hessian_N_list, results_hessians[:,method['algopy_forward_utpm'],0], '-.k>', markerfacecolor='None', label = 'algopy (fo)')
pyplot.loglog(hessian_N_list, results_hessians[:,method['algopy_reverse_utpm'],0], '-.k<', markerfacecolor='None', label = 'algopy (fo/rev)')
pyplot.loglog(hessian_N_list, results_hessians[:,method['numdifftools'],0], '--ks', markerfacecolor='None', label = 'numdifftools')
pyplot.ylabel('time $t$ [seconds]')
pyplot.xlabel('problem size $N$')
pyplot.grid()
leg = pyplot.legend(loc=2)
frame= leg.get_frame()
frame.set_alpha(0.4)
pyplot.savefig('hessian_runtimes.png')

pyplot.show()



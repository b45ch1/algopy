import numpy as np
import time

import benchmark1
import use_adolc
import use_scientific
import use_uncertainties
import use_algopy
import use_numdifftools
import use_funcdesigner
import use_theano


method = {'algopy_reverse_utps':0, 'algopy_reverse_utpm':1, 'pyadolc':2, 'scientific':3, 'uncertainties':4, 'numdifftools':5, 'algopy_forward_utpm':6, 'python':7, 'algopy_forward_utps':8, 'funcdesigner':9, 'theano':10}


# FUNCTION COMPUTATION
# --------------------

function_N_list = [1,2,4,8,16,32,64,128,256]
# function_N_list = [2]


results_function_list = []
for N in function_N_list:
    print('N=',N)
    results_function = np.zeros((11,3))
     
    # pure python
    f = benchmark1.F(N)
    t = time.time(); pass ; preproc_time = time.time() - t
    t = time.time();  ref_f =  f(3*np.ones(N));  run_time = time.time() - t
    results_function[method['python']] = run_time, abs(ref_f - ref_f), preproc_time
    print('ref_f=',ref_f)
    
    # pyadolc
    f = benchmark1.F(N)
    t = time.time();  a = use_adolc.EVAL(f, np.ones(N), test='f'); preproc_time = time.time() - t
    t = time.time();  f =  a.function(3*np.ones(N));  run_time = time.time() - t
    results_function[method['pyadolc']] = run_time, abs(ref_f - f), preproc_time
    
    # algopy utps version
    f = benchmark1.F(N)
    t = time.time();  a = use_algopy.EVAL(f, np.ones(N), test='f'); preproc_time = time.time() - t
    t = time.time();  f =  a.function(3*np.ones(N));  run_time = time.time() - t
    results_function[method['algopy_forward_utps']] = run_time, abs(ref_f - f), preproc_time    
    
    # algopy utpm version
    f = benchmark1.G(N)
    t = time.time();  a = use_algopy.EVAL2(f, np.ones(N), test='f'); preproc_time = time.time() - t
    t = time.time();  f =  a.function(3*np.ones(N));  run_time = time.time() - t
    results_function[method['algopy_forward_utpm']] = run_time, abs(ref_f - f), preproc_time        

    # funcdesigner
    f = benchmark1.F(N)
    t = time.time();  a = use_funcdesigner.EVAL(f, np.ones(N), test='f'); preproc_time = time.time() - t
    t = time.time();  f =  a.function(3*np.ones(N));  run_time = time.time() - t
    results_function[method['funcdesigner']] = run_time, abs(ref_f - f), preproc_time
    
    # theano
    f = benchmark1.F(N)
    t = time.time();  a = use_theano.EVAL(f, np.ones(N), test='f'); preproc_time = time.time() - t
    t = time.time();  f =  a.function(3*np.ones(N));  run_time = time.time() - t
    results_function[method['theano']] = run_time, np.abs(ref_f - f), preproc_time        
    results_function_list.append(results_function)
    # print f

results_functions = np.array(results_function_list)
print('results_functions=\n',results_functions)

# GRADIENT COMPUTATION
# --------------------

gradient_N_list = [2,4,8,16,32,64,96]
# gradient_N_list = [20]

results_gradient_list = []
for N in gradient_N_list:
    print('N=',N)
    results_gradient = np.zeros((11,3))
    
    # pyadolc
    f = benchmark1.F(N)
    t = time.time();  a = use_adolc.EVAL(f, np.ones(N), test='g'); preproc_time = time.time() - t
    t = time.time();  ref_g =  a.gradient(3*np.ones(N));  run_time = time.time() - t
    results_gradient[method['pyadolc']] = run_time, 0, preproc_time
    
    # scientifc
    f = benchmark1.F(N)
    t = time.time();  a = use_scientific.EVAL(f, np.ones(N), test='g'); preproc_time = time.time() - t
    t = time.time();  g =  a.gradient(3*np.ones(N));  run_time = time.time() - t
    results_gradient[method['scientific']] = run_time,  np.linalg.norm(g - ref_g)/np.linalg.norm(ref_g), preproc_time
    
    # algopy, UTPS variant
    f = benchmark1.F(N)
    t = time.time();  a = use_algopy.EVAL(f, np.ones(N), test='g'); preproc_time = time.time() - t
    t = time.time();  g =  a.gradient(3*np.ones(N));  run_time = time.time() - t
    results_gradient[method['algopy_reverse_utps']] = run_time,  np.linalg.norm(g - ref_g)/np.linalg.norm(ref_g), preproc_time
        
    # algopy, UTPM variant
    f = benchmark1.G(N)
    t = time.time();  a = use_algopy.EVAL2(f, np.ones(N), test='g'); preproc_time = time.time() - t
    t = time.time();  g =  a.gradient(3*np.ones(N));  run_time = time.time() - t
    results_gradient[method['algopy_reverse_utpm']] = run_time,  np.linalg.norm(g - ref_g)/np.linalg.norm(ref_g), preproc_time
    
    # algopy, forward UTPM variant
    f = benchmark1.G(N)
    t = time.time();  a = use_algopy.EVAL2(f, np.ones(N), test='fg'); preproc_time = time.time() - t
    t = time.time();  g = a.forwardgradient(3*np.ones(N));  run_time = time.time() - t
    results_gradient[method['algopy_forward_utpm']] = run_time,  np.linalg.norm(g - ref_g)/np.linalg.norm(ref_g), preproc_time
    
    # numdifftools
    f = benchmark1.F(N)
    t = time.time();  a = use_numdifftools.EVAL(f, np.ones(N), test='g'); preproc_time = time.time() - t
    t = time.time();  g =  a.gradient(3*np.ones(N));  run_time = time.time() - t
    results_gradient[method['numdifftools']] = run_time,  np.linalg.norm(g - ref_g)/np.linalg.norm(ref_g), preproc_time
      
    # uncertainties
    f = benchmark1.F(N)
    t = time.time();  a = use_uncertainties.EVAL(f, np.ones(N), test='g'); preproc_time = time.time() - t
    t = time.time();  g =  a.gradient(3*np.ones(N));  run_time = time.time() - t
    results_gradient[method['uncertainties']] = run_time,  np.linalg.norm(g - ref_g)/np.linalg.norm(ref_g), preproc_time
    
    # funcdesigner
    f = benchmark1.F(N)
    t = time.time();  a = use_funcdesigner.EVAL(f, np.ones(N), test='g'); preproc_time = time.time() - t
    t = time.time();  g =  a.gradient(3*np.ones(N));  run_time = time.time() - t
    results_gradient[method['funcdesigner']] = run_time,  np.linalg.norm(g - ref_g)/np.linalg.norm(ref_g), preproc_time
    
    # theano
    f = benchmark1.F(N)
    t = time.time();  a = use_theano.EVAL(f, np.ones(N), test='g'); preproc_time = time.time() - t
    t = time.time();  g =  a.gradient(3*np.ones(N));  run_time = time.time() - t
    results_gradient[method['theano']] = run_time,  np.linalg.norm(g - ref_g)/np.linalg.norm(ref_g), preproc_time
    
    results_gradient_list.append(results_gradient)

results_gradients = np.array(results_gradient_list)
print('results_gradients=\n',results_gradients)

# HESSIAN COMPUTATION
# -------------------
print('starting hessian computation ')
results_hessian_list = []
hessian_N_list = [1,2,4,8,16,32,64]
# hessian_N_list = [2]

for N in hessian_N_list:
    print('N=',N)
    results_hessian = np.zeros((11,3))
    
    # pyadolc
    f = benchmark1.F(N)
    t = time.time();  a = use_adolc.EVAL(f, np.ones(N), test='h'); preproc_time = time.time() - t
    t = time.time();  ref_H =  a.hessian(3*np.ones(N));  run_time = time.time() - t
    results_hessian[method['pyadolc']] = run_time, 0, preproc_time
    
    # numdifftools
    f = benchmark1.F(N)
    t = time.time();  a = use_numdifftools.EVAL(f, np.ones(N), test='h'); preproc_time = time.time() - t
    t = time.time();  H =  a.hessian(3*np.ones(N));  run_time = time.time() - t
    results_hessian[method['numdifftools']] = run_time, np.linalg.norm( (H-ref_H).ravel())/ np.linalg.norm( (ref_H).ravel()), preproc_time
    
    # algopy forward utpm variant
    f = benchmark1.G(N)
    t = time.time();  a = use_algopy.EVAL2(f, np.ones(N), test='fh'); preproc_time = time.time() - t
    t = time.time();  H = a.forwardhessian(3*np.ones(N));  run_time = time.time() - t
    results_hessian[method['algopy_forward_utpm']] = run_time, np.linalg.norm( (H-ref_H).ravel())/ np.linalg.norm( (ref_H).ravel()), preproc_time
    
    # algopy forward/reverse utpm variant
    f = benchmark1.G(N)
    t = time.time();  a = use_algopy.EVAL2(f, np.ones(N), test='h'); preproc_time = time.time() - t
    t = time.time();  H = a.hessian(3*np.ones(N));  run_time = time.time() - t
    results_hessian[method['algopy_reverse_utpm']] = run_time, np.linalg.norm( (H-ref_H).ravel())/ np.linalg.norm( (ref_H).ravel()), preproc_time    
    
    # theano
    f = benchmark1.F(N)
    t = time.time();  a = use_theano.EVAL(f, np.ones(N), test='h'); preproc_time = time.time() - t
    t = time.time();  H = a.hessian(3*np.ones(N));  run_time = time.time() - t
    results_hessian[method['theano']] = run_time, np.linalg.norm( (H-ref_H).ravel())/ np.linalg.norm( (ref_H).ravel()), preproc_time
    
    results_hessian_list.append(results_hessian)
    

results_hessians = np.array(results_hessian_list)

print(hessian_N_list)
print('results_hessians=\n',results_hessians)


# PLOT RESULTS

print(results_gradients.shape)

import matplotlib.pyplot as pyplot
import prettyplotting

# plot function run times
pyplot.figure()
pyplot.title('Function run times')
pyplot.loglog(function_N_list, results_functions[:,method['pyadolc'],0], '-ko', markerfacecolor='None', label = 'pyadolc')
pyplot.loglog(function_N_list, results_functions[:,method['theano'],0], '-.k1', markerfacecolor='None', label = 'theano')

pyplot.loglog(function_N_list, results_functions[:,method['python'],0], '-ks', markerfacecolor='None', label = 'python')
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
pyplot.loglog(gradient_N_list, results_gradients[:,method['theano'],0], '-.k1', markerfacecolor='None', label = 'theano')
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
pyplot.loglog(hessian_N_list, results_hessians[:,method['theano'],0], '-.k1', markerfacecolor='None', label = 'theano')
pyplot.ylabel('time $t$ [seconds]')
pyplot.xlabel('problem size $N$')
pyplot.grid()
leg = pyplot.legend(loc=2)
frame= leg.get_frame()
frame.set_alpha(0.4)
pyplot.savefig('hessian_runtimes.png')


# plot gradient preprocessing times
pyplot.figure()
pyplot.title('Gradient preprocessing times')
pyplot.loglog(gradient_N_list, results_gradients[:,method['pyadolc'],2], '-ko', markerfacecolor='None', label = 'pyadolc')
pyplot.loglog(gradient_N_list, results_gradients[:,method['algopy_reverse_utps'],2], '-.k<', markerfacecolor='None', label = 'algopy reverse utps')
pyplot.loglog(gradient_N_list, results_gradients[:,method['algopy_reverse_utpm'],2], '-.k>', markerfacecolor='None', label = 'algopy reverse utpm')
pyplot.loglog(gradient_N_list, results_gradients[:,method['algopy_forward_utpm'],2], '-.kv', markerfacecolor='None', label = 'algopy forward utpm')
pyplot.loglog(gradient_N_list, results_gradients[:,method['uncertainties'],2], '--kh', markerfacecolor='None', label = 'uncertainties')
pyplot.loglog(gradient_N_list, results_gradients[:,method['numdifftools'],2], '--ks', markerfacecolor='None', label = 'numdifftools')
pyplot.loglog(gradient_N_list, results_gradients[:,method['funcdesigner'],2], '--kd', markerfacecolor='None', label = 'funcdesigner')
pyplot.loglog(gradient_N_list, results_gradients[:,method['theano'],2], '-.k1', markerfacecolor='None', label = 'theano')
pyplot.ylabel('time $t$ [seconds]')
pyplot.xlabel('problem size $N$')
pyplot.grid()
leg = pyplot.legend(loc=2)
frame= leg.get_frame()
frame.set_alpha(0.4)
pyplot.savefig('gradient_preprocessingtimes.png')

# plot hessian preprocessing times
pyplot.figure()
pyplot.title('Hessian preprocessing times')
pyplot.loglog(hessian_N_list, results_hessians[:,method['pyadolc'],2], '-ko', markerfacecolor='None', label = 'pyadolc (fo/rev)')
pyplot.loglog(hessian_N_list, results_hessians[:,method['algopy_forward_utpm'],2], '-.k>', markerfacecolor='None', label = 'algopy (fo)')
pyplot.loglog(hessian_N_list, results_hessians[:,method['algopy_reverse_utpm'],2], '-.k<', markerfacecolor='None', label = 'algopy (fo/rev)')
pyplot.loglog(hessian_N_list, results_hessians[:,method['numdifftools'],2], '--ks', markerfacecolor='None', label = 'numdifftools')
pyplot.loglog(hessian_N_list, results_hessians[:,method['theano'],2], '-.k1', markerfacecolor='None', label = 'theano')
pyplot.ylabel('time $t$ [seconds]')
pyplot.xlabel('problem size $N$')
pyplot.grid()
leg = pyplot.legend(loc=2)
frame= leg.get_frame()
frame.set_alpha(0.4)
pyplot.savefig('hessian_preprocessingtimes.png')

# plot gradient errors
pyplot.figure()
pyplot.title('Gradient Correctness')
pyplot.loglog(gradient_N_list, results_gradients[:,method['numdifftools'],1], '--ks', markerfacecolor='None', label = 'numdifftools')
pyplot.loglog(gradient_N_list, results_gradients[:,method['pyadolc'],1], '-ko', markerfacecolor='None', label = 'pyadolc')
pyplot.loglog(gradient_N_list, results_gradients[:,method['algopy_reverse_utps'],1], '-.k<', markerfacecolor='None', label = 'algopy reverse utps')
pyplot.loglog(gradient_N_list, results_gradients[:,method['algopy_reverse_utpm'],1], '-.k>', markerfacecolor='None', label = 'algopy reverse utpm')
pyplot.loglog(gradient_N_list, results_gradients[:,method['algopy_forward_utpm'],1], '-.kv', markerfacecolor='None', label = 'algopy forward utpm')
pyplot.loglog(gradient_N_list, results_gradients[:,method['uncertainties'],1], '--kh', markerfacecolor='None', label = 'uncertainties')
pyplot.loglog(gradient_N_list, results_gradients[:,method['funcdesigner'],1], '--kd', markerfacecolor='None', label = 'funcdesigner')
pyplot.loglog(gradient_N_list, results_gradients[:,method['theano'],1], '-.k1', markerfacecolor='None', label = 'theano')
pyplot.ylabel(r'relative error $\|g_{ref} - g\|/\|g_{ref}\}$')
pyplot.xlabel('problem size $N$')
pyplot.grid()
leg = pyplot.legend(loc=2)
frame= leg.get_frame()
frame.set_alpha(0.4)
pyplot.savefig('gradient_errors.png')

# plot hessian errors
pyplot.figure()
pyplot.title('Hessian Correctness')
pyplot.loglog(hessian_N_list, results_hessians[:,method['numdifftools'],1], '--ks', markerfacecolor='None', label = 'numdifftools')
pyplot.loglog(hessian_N_list, results_hessians[:,method['pyadolc'],1], '-ko', markerfacecolor='None', label = 'pyadolc (fo/rev)')
pyplot.loglog(hessian_N_list, results_hessians[:,method['algopy_forward_utpm'],1], '-.k>', markerfacecolor='None', label = 'algopy (fo)')
pyplot.loglog(hessian_N_list, results_hessians[:,method['algopy_reverse_utpm'],1], '-.k<', markerfacecolor='None', label = 'algopy (fo/rev)')
pyplot.loglog(hessian_N_list, results_hessians[:,method['theano'],1], '-.k1', markerfacecolor='None', label = 'theano')
pyplot.ylabel(r'relative error $\|H_{ref} - H\|/\|H_{ref}\|$')
pyplot.xlabel('problem size $N$')
pyplot.grid()
leg = pyplot.legend(loc=2)
frame= leg.get_frame()
frame.set_alpha(0.4)
pyplot.savefig('hessian_preprocessingtimes.png')



# pyplot.show()



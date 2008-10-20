#!/usr/bin/env python
import sys
import numpy
import numpy.linalg
from  time import time


sys.path = ['..'] + sys.path
import vector_forward_mode as vfm
import reverse_mode as rm
import adolc


Ns = [2**i for i in range(11)]
function_eval_times = []
forward_eval_times = []
tape_rev_eval_times = []
tape_eval_times = []
rev_eval_times = []
adolc_tape_times = []
adolc_gradient_times = []



for N in Ns:
	print N
	#A = 2.3*numpy.eye(N)
	x = numpy.ones(N)

	#def f(x):
		#return numpy.dot(x, numpy.dot(A,x))

	def f(x):
		return numpy.sum(x*x)


	## normal function evaluation
	start_time = time()
	y = f(x)
	end_time = time()
	function_eval_times.append(end_time-start_time)


	## vector forward evaluation
	start_time = time()
	g_forward = vfm.vector_gradient(f,x)
	end_time = time()
	forward_eval_times.append(end_time-start_time)


	## taping + reverse evaluation
	start_time = time()
	g_reverse = rm.gradient(f,x)
	end_time = time()
	tape_rev_eval_times.append(end_time-start_time)


	## taping
	start_time = time()
	cg = rm.tape(f,x)
	end_time = time()
	tape_eval_times.append(end_time-start_time)

	## reverse evaluation
	start_time = time()
	g_reverse2 = rm.gradient_from_graph(cg)
	end_time = time()
	rev_eval_times.append(end_time-start_time)

	## PyADOLC taping
	start_time = time()
	ax = numpy.array([adolc.adouble(0.) for i in range(N)])
	adolc.trace_on(0)
	for n in range(N):
		ax[n].is_independent(x[n])
	ay = f(ax)
	adolc.depends_on(ay)
	adolc.trace_off()
	end_time = time()
	adolc_tape_times.append(end_time-start_time)
	
	## PyADOLC gradient
	start_time = time()
	adolc_g = adolc.gradient(0,x)
	end_time = time()
	adolc_gradient_times.append(end_time-start_time)


	### check that both derivatives give the same result
	#print 'difference between forward and reverse gradient computation', numpy.linalg.norm(g_forward - g_reverse)
	#print 'difference between forward and reverse gradient2 computation', numpy.linalg.norm(g_forward - g_reverse2)
	#print 'difference between Algopy and PyAdolc', numpy.linalg.norm(adolc_g - g_reverse2)



import pylab

function_plot        = pylab.loglog(Ns,function_eval_times, 'r.')
forward_plot         = pylab.loglog(Ns,forward_eval_times, 'b.')
taperev_plot         = pylab.loglog(Ns,tape_rev_eval_times, 'r^')
tape_plot            = pylab.loglog(Ns,tape_eval_times, 'b^')
rev_plot             = pylab.loglog(Ns,rev_eval_times, 'cs')
adolc_tape_plot      = pylab.loglog(Ns,adolc_tape_times, 'rd')
adolc_gradient_plot  = pylab.loglog(Ns,adolc_gradient_times, 'bd')

pylab.xlabel('number of independent variables N []')
pylab.ylabel('runtime t [sec]')
pylab.grid()

pylab.legend((function_plot,forward_plot,taperev_plot,tape_plot,rev_plot,adolc_tape_plot,adolc_gradient_plot), ('function','forward', 'tape+rev', 'tape', 'rev','adolc tape', 'adolc gradient'), loc=2)
pylab.savefig('runtime_comparison.eps')
pylab.show()




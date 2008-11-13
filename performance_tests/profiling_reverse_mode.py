from numpy import *
import pstats
import profile
import sys

sys.path = ['..'] + sys.path
from reverse_mode import *

def profile_taylor_propagation():
	N = 20
	A = ones((N,N))
	x = [Tc(n) for n in range(N)]
	y = dot(A,x)

def profile_reverse():
	N = 20
	A = ones((N,N))
	def fun(x):
		return 0.5* dot(x, dot(A,x))
	
	cg = CGraph()
	x = [Function(Tc([1.,1.,0.])) for n in range(N)]
	g = fun(x)
	cg.independentFunctionList = x
	cg.dependentFunctionList = [g]
	cg.reverse([Tc(1)])
	cg.reverse([Tc(1)])
	cg.reverse([Tc(1)])


if __name__ == "__main__":
	profile.run('profile_reverse()', 'profile_reverse_stats')
	p = pstats.Stats('profile_reverse_stats')
	p.strip_dirs().sort_stats('cumulative').print_stats()

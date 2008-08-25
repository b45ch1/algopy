#!/usr/bin/env python

from scipy import *
from numpy import *



# 2D case
N = 4
D = 2
M = factorial(N+D-1)/(factorial(D) * factorial(N-1))

T = zeros((M,N),dtype=int)
pos = 0
for n1 in range(N):
	for n2 in range(n1,N):
		t = zeros(N)
		t[n1]+=1
		t[n2]+=1
		T[pos,:] = t
		pos+=1

print array(T).__repr__()

# 3D case

N = 4
D = 3
M = factorial(N+D-1)/(factorial(D) * factorial(N-1))

T = zeros((M,N),dtype=int)
pos = 0
for n1 in range(N):
	for n2 in range(n1,N):
		for n3 in range(n2,N):
			t = zeros(N)
			t[n1]+=1
			t[n2]+=1
			t[n3]+=1
			T[pos,:] = t
			pos+=1

print array(T).__repr__()


# ND case
N = 4
D = 3
M = int(factorial(N+D-1)/(factorial(D) * factorial(N-1)))

print 'N,D,M=',N,D,M
T = []
def rec(r,n,N,D):
	j = r.copy()
	if n == N-1:
		j[N-1] = D - sum(j[:])
		T.append(j.copy())
		return
	for a in range( D - sum( j [:] ), -1,-1 ):
		j[n]=a
		rec(j,n+1,N,D)

r = zeros(N,dtype=int)

rec(r,0,N,D)

print array(T).__repr__()


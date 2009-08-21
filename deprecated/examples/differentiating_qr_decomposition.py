from adolc import *
import adolc
from numpy import *

def myqr(in_A):
	# input checks
	Ndim = ndim(in_A)
	assert Ndim == 2
	N,M = shape(in_A)
	assert N==M

	# algorithm
	R = in_A.copy()
	if type(R[0,0]) == adolc._adolc.adouble:
		QT = array([[adouble(n==m) for m in range(N)] for n in range(N) ])
	else:
		QT = eye(N)

	for n in range(N):
		for m in range(n+1,N):
			a = R[n,n]
			b = R[m,n]
			r = sqrt(a**2 + b**2)
			c = a/r
			s = b/r

			for k in range(N):
				Rnk = R[n,k]
	
				R[n,k] = c*Rnk + s*R[m,k]
				R[m,k] =-s*Rnk + c*R[m,k];

				QTnk = QT[n,k]
				QT[n,k] = c*QTnk + s*QT[m,k]
				QT[m,k] =-s*QTnk + c*QT[m,k]
			#print 'QT:\n',QT
			#print 'R:\n',R
			#print '-------------'

	return QT,R
			
def inv(in_A):
	QT,R = myqr(in_A)
	N = shape(in_A)[0]

	for n in range(N-1,-1,-1):
		Rnn = R[n,n]
		R[n,:] /= Rnn
		QT[n,:] /= Rnn
		for m in range(n+1,N):
			Rnm = R[n,m]
			R[n,m] = 0
			QT[n,:] -= QT[m,:]*Rnm

	return QT,R


if __name__ == "__main__":
	N = 4
	A = array([[ r*N + c for c in range(N) ] for r in range(N)],dtype=float)

	aA = adouble(A)
	trace_on(1)
	independent(aA)
	(aQT, aR) = myqr(aA)
	dependent(aQT)
	dependent(aR)
	trace_off()

	V = zeros((N**2,1,1),dtype=float)
	V[:,0,0] = 1.
	(result,Z) = hov_forward(1, ravel(A), V)
	QT = result[:N**2].reshape((N,N))
	R  = result[N**2:].reshape((N,N))

	#(QT,R) = myqr(A)
	#print dot(QT.T,R) - A

	QTdot = Z[:N**2,0,0].reshape((N,N))
	Rdot  = Z[N**2:,0,0].reshape((N,N))

	print QTdot.T + dot(dot(QT.T,QTdot), QT.T)

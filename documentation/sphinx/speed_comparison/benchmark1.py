import numpy as np
import algopy

class F:
    def __init__(self,N):
        A = np.arange(N*N,dtype=float).reshape((N,N))
        self.A = np.dot(A.T,A)

    def __call__(self, x):
        return 0.5*np.dot(x*x,np.dot(self.A,x))
        
        
class G:
    def __init__(self,N):
        A = np.arange(N*N,dtype=float).reshape((N,N))
        self.A = np.dot(A.T,A)

    def __call__(self, x):
        return 0.5*algopy.dot(x*x,algopy.dot(self.A,x))

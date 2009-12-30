from numpy.testing import *
import numpy

from algopy.utp.utpm import *


from algopy.utp.tests.utpm.test_utpm_class_functions import *
from algopy.utp.tests.utpm.test_utpm_instance_functions import *
from algopy.utp.tests.utpm.test_utpm_global_functions import *


class ODOE_example_for_ICCS2010_conference(TestCase):
    def test_forward(self):
        D,Nx,Ny,NF = 2,3,3,20
        P = Ny
        x = UTPM(numpy.random.rand(D,P,Nx))
        y = UTPM(numpy.random.rand(D,P,Ny))

        F = UTPM(numpy.zeros((D,P,NF)))

        for nf in range(NF):
            F[nf] =  numpy.sum([ (nf+1.)**n * x[n]*y[-n] for n in range(Nx)])
        
        J = F.FtoJT().T
        Q,R = UTPM.qr(J)

        Id = numpy.eye(P)
        D = UTPM.solve(R.T,Id)
        C = UTPM.solve(D,R)
        l,U = UTPM.eig(C)

        l11 = l.max()
        
        print l11




if __name__ == "__main__":
    run_module_suite()

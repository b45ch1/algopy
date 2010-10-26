Least-Squares Fitting
---------------------

This is an example from the Scipy-User mailing list. The original discussion 
can be found on
http://permalink.gmane.org/gmane.comp.python.scientific.user/26551 .

.. literalinclude:: leastsquaresfitting.py
    :lines: 0-


We provide the Jacobian of the error function using ALGOPY and compare
    1) the solution when scipy.optimize.leastsq approximates the Jacobian with finite differences
    2) when the Jacobian is provided to scipy.optimize.leastsq

As output one obtains::
    $ python leastsquaresfitting.py 
    Estimates from leastsq 
    [  6.79548889e-02   3.68922501e-01   7.55565769e-02   1.41378227e+02
       2.91307741e+00   2.70608242e+02] 1
    number of function calls = 26
    Estimates from leastsq 
    [  6.79548883e-02   3.68922503e-01   7.55565728e-02   1.41378227e+02
       2.91307814e+00   2.70608242e+02] 1
    number of function calls = 140


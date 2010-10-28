Hessian of Two Particle Coulomb Potential
-----------------------------------------

This is an example from the Scipy-User mailing list. The original discussion 
can be found on
http://www.mail-archive.com/numpy-discussion@scipy.org/msg28633.html

.. literalinclude:: hessian_of_potential_function.py
    :lines: 0-
    
    
As output one obtains::
    
    $ python hessian_of_potential_function.py 
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 81
             Function evaluations: 153
    [[  5.23748399e-12  -2.61873843e-12]
     [ -2.61873843e-12   5.23748399e-12]]



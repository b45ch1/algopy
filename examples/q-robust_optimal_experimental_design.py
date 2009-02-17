#!/usr/bin/env python
from pylab import *
import sys
sys.path = ['..'] + sys.path
from matrix_ad import *
import adolc
import numpy.random
import scipy.optimize

from prettyplotting import * # comment this out if not available


"""
# We look at the following q-robust OED problem:
# variables:
#        p = v[:Np] = parameters, Np is number of parameters
#        q = v[Np:] = control variables
#
# model: ODE
#        dx/dt = f(t,x,p) = p0
#        x(0)  = x_0 = p1
#
# measurement model:
#        h(t,x,v) = x(t,v)
# parameter estimation:
#        F(p) = [F1, ..., FM].T
#        F = Sigma^-1 ( eta - h(t,x,p,q) )
#        eta are the measurements
#        p_   = argmin_p |F(p)|_2^2
# q-robust OED:
#        J = dF/dp
#        C = (J^T J)^-1  covariance matrix of the parameters
#        Phi(q) = tr(C(q))
#     /---------------------------------------------\
#     |   q_ = argmin_{mu \in MU} E_mu[Phi(Q)]       |
#     \---------------------------------------------/
#        where mu is a probability measure
#        MU is a family of probability distributions
#        here, MU := {mu : mu uniformly distributed in q-sigma, q + sigma}
"""





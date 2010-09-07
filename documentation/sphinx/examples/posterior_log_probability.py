import numpy; from algopy import UTPM, zeros

def LogNormalLikelihood(x, mu, sigma):
   return sigma *(x - mu)**2  - numpy.log(.5*sigma/numpy.pi)

def logp(x, mu, sigma):
    return numpy.sum(LogNormalLikelihood(x, mu, sigma)) + LogNormalLikelihood(mu , mu_prior_mean, mu_prior_sigma)

mu_prior_mean = 1
mu_prior_sigma = 5

actual_mu = 3.1
sigma = 1.2
N = 35
x = numpy.random.normal(actual_mu, sigma, size = N)
mu = UTPM([[3.5],[1]]) #unknown variable

print 'function evaluation =\n',logp(x,3.5,sigma)
print 'function evaluation + 1st directional derivative =\n',logp(x, mu, sigma)

# finite differences solution:
print 'finite differences derivative =\n',(logp(x,3.5+10**-8,sigma) - logp(x, 3.5, sigma))/10**-8



from algopy.tracer.tests.test_tracer import *
from algopy.utps.tests.test_utps import *
from algopy.utpm.tests.test_utpm import *
from algopy.utpm.tests.test_algorithms import *
from algopy.tests.test_globalfuncs import *

if __name__ == '__main__':
    try:
        import nose
    except:
        print 'Please install nose for unit testing'
    nose.runmodule()


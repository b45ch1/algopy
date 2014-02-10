from algopy.tracer.tests.test_tracer import *
from algopy.utpm.tests.test_utpm import *
from algopy.utpm.tests.test_utpm_convenience import *
from algopy.utpm.tests.test_algorithms import *
from algopy.tests.test_globalfuncs import *
from algopy.tests.test_exact_interpolation import *
from algopy.tests.test_utils import *
from algopy.tests.test_npversion import *
from algopy.tests.test_compound import *


if __name__ == '__main__':
    try:
        import nose
    except:
        print('Please install nose for unit testing')
    nose.runmodule()


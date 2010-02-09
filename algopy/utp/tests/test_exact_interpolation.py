"""
Test algpy.utp.exact_interpolation

Not working ATM.
"""

from numpy.testing import *
import numpy

from algopy.utp.exact_interpolation import *


class TestExactInterpolation(TestCase):
    def test_generate_multi_indices(self):
        a = numpy.array(
            [[3, 0, 0, 0],
            [2, 1, 0, 0],
            [2, 0, 1, 0],
            [2, 0, 0, 1],
            [1, 2, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 2, 0],
            [1, 0, 1, 1],
            [1, 0, 0, 2],
            [0, 3, 0, 0],
            [0, 2, 1, 0],
            [0, 2, 0, 1],
            [0, 1, 2, 0],
            [0, 1, 1, 1],
            [0, 1, 0, 2],
            [0, 0, 3, 0],
            [0, 0, 2, 1],
            [0, 0, 1, 2],
            [0, 0, 0, 3]])
        assert_array_equal(generate_multi_indices(4,3), a)

    def test_convert_multi_indices_to_pos(self):
        N,D = 4,3
        I = generate_multi_indices(N,D)
        computed_pos_mat = convert_multi_indices_to_pos(I)
        true_pos_mat = numpy.array([[0, 0, 0],
        [0, 0, 1],
        [0, 0, 2],
        [0, 0, 3],
        [0, 1, 1],
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 2],
        [0, 2, 3],
        [0, 3, 3],
        [1, 1, 1],
        [1, 1, 2],
        [1, 1, 3],
        [1, 2, 2],
        [1, 2, 3],
        [1, 3, 3],
        [2, 2, 2],
        [2, 2, 3],
        [2, 3, 3],
        [3, 3, 3]], dtype=int)
        #print true_pos_mat
        #print computed_pos_mat
        assert numpy.prod(true_pos_mat == computed_pos_mat) #all entries have to be the same

    def test_generate_permutations(self):
        x = [1,2,3]
        computed_perms = []
        for p in generate_permutations(x):
            computed_perms += [p]
        computed_perms = numpy.array(computed_perms)
        true_perms = numpy.array([[1, 2, 3],[2, 1, 3],[2, 3, 1],[1, 3, 2],[3, 1, 2],[3, 2, 1]],dtype=int)
        assert numpy.prod(computed_perms == true_perms)

    def test_generate_Gamma(self):
        i = numpy.array([1,1],dtype=int)
        (Gamma,J) = generate_Gamma(i)
        assert_array_almost_equal(Gamma, [ - 0.25, 1., -0.25])

    # def test_tensor(self):
    #     def f(x):
    #         return numpy.prod(x)
    #     x = numpy.array([1.,2.,3.])
    #     computed_tensor = vector_tensor(f,x,3)

    #     true_tensor = numpy.array([[[ 0.,  0.,  0.],
    # [ 0.,  0.,  1.],
    # [ 0.,  1.,  0.]],

    # [[ 0.,  0.,  1.],
    # [ 0.,  0.,  0.],
    # [ 1.,  0.,  0.]],

    # [[ 0.,  1.,  0.],
    # [ 1.,  0.,  0.],
    # [ 0.,  0.,  0.]]])
    #     print 'true_tensor=', true_tensor
    #     print 'computed_tensor=', computed_tensor
    #     assert numpy.prod(computed_tensor == true_tensor)

    # def test_vector_hessian(self):
    #     import time
    #     def f(x):
    #         return numpy.prod(x)
    #     x = numpy.array([i+1 for i in range(10)])
    #     start_time = time.time()
    #     computed_hessian = vector_hessian(f,x)
    #     end_time = time.time()
    #     print computed_hessian
    #     print 'run time=%0.6f seconds'%(end_time-start_time)
    #     assert True



if __name__ == "__main__":
    run_module_suite()
 

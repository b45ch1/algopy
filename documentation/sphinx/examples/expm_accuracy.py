"""
Test matrix exponential functions.

This includes expm test matrices
collected by John Burkardt from various sources as of Oct 2012.
The last of his 13 test cases is not included, because it is parameterized.
"""

import math

import numpy

import algopy


# Use the correct eps when understanding Matlab notation.
# http://www.scipy.org/NumPy_for_Matlab_Users
matlab_eps = numpy.spacing(1)

# More constants for the expm test matrices.
exp16 = numpy.exp(16)
exp4 = numpy.exp(4)


class ExpmCase:
    def __init__(self, burkardt_index, description, M, expm_M):
        self.burkardt_index = burkardt_index
        self.description = description
        self.M = M
        self.expm_M = expm_M

g_burkardt_expm_cases = [

        ExpmCase(
            -1,
            'This is from a blog post by Cleve Moler.\n'
            'It plays poorly with Higham 2005 scaling.\n'
            'It is not yet collected by Burkardt hence the -1 index.',
            numpy.array([
                [0, 1e-8, 0],
                [-(2e10 + 4e8/6.), -3, 2e10],
                [200./3., 0, -200./3.],
                ], dtype=float),
            numpy.array([
                [0.446849468283175, 1.54044157383952e-09, 0.462811453558774],
                [-5743067.77947947, -0.0152830038686819, -4526542.71278401],
                [0.447722977849494, 1.54270484519591e-09, 0.463480648837651],
                ], dtype=float),
            ),

        ExpmCase(
            1,
            'This matrix is diagonal.\n'
            'The calculation of the matrix exponential is simple.',
            numpy.array([
                [1, 0],
                [0, 2],
                ], dtype=float),
            numpy.array([
                [2.718281828459046, 0],
                [0, 7.389056098930650],
                ], dtype=float),
            ),

        ExpmCase(
            2,
            'This matrix is symmetric.\n'
            'The calculation of the matrix exponential is straightforward.',
            numpy.array([
                [1, 3],
                [3, 2],
                ], dtype=float),
            numpy.array([
                [39.322809708033859, 46.166301438885753],
                [46.166301438885768, 54.711576854329110],
                ], dtype=float),
            ),

        ExpmCase(
            3,
            'This example is due to Laub.\n'
            'This matrix is ill-suited for the Taylor series approach.\n'
            'As powers of A are computed, the entries blow up too quickly.',
            numpy.array([
                [0, 1],
                [-39, -40],
                ], dtype=float),
            numpy.array([
                # The input matrix is correct in the Laub reference,
                # but the exact solution was somehow entered incorrectly.
                #[0, 2.718281828459046],
                #[1.154822e-17, 2.718281828459046],
                [
                    39/(38*math.exp(1)) - 1/(38*math.exp(39)),
                    -math.expm1(-38) / (38*math.exp(1))],
                [
                    39*math.expm1(-38) / (38*math.exp(1)),
                    -1/(38*math.exp(1)) + 39/(38*math.exp(39))],
                ], dtype=float),
            ),

        ExpmCase(
            4,
            'This example is due to Moler and Van Loan.\n'
            'The example will cause problems '
            'for the series summation approach,\n'
            'as well as for diagonal Pade approximations.',
            numpy.array([
                [-49, 24],
                [-64, 31],
                ], dtype=float),
            numpy.array([
                [-0.735759, 0.551819],
                [-1.471518, 1.103638],
                ], dtype=float),
            ),

        ExpmCase(
            5,
            'This example is due to Moler and Van Loan.\n'
            'This matrix is strictly upper triangular\n'
            'All powers of A are zero beyond some (low) limit.\n'
            'This example will cause problems for Pade approximations.',
            numpy.array([
                [0, 6, 0, 0],
                [0, 0, 6, 0],
                [0, 0, 0, 6],
                [0, 0, 0, 0],
                ], dtype=float),
            numpy.array([
                [1, 6, 18, 36],
                [0, 1, 6, 18],
                [0, 0, 1, 6],
                [0, 0, 0, 1],
                ], dtype=float),
            ),

        ExpmCase(
            6,
            'This example is due to Moler and Van Loan.\n'
            'This matrix does not have a complete set of eigenvectors.\n'
            'That means the eigenvector approach will fail.',
            numpy.array([
                [1, 1],
                [0, 1],
                ], dtype=float),
            numpy.array([
                [2.718281828459046, 2.718281828459046],
                [0, 2.718281828459046],
                ], dtype=float),
            ),

        ExpmCase(
            7,
            'This example is due to Moler and Van Loan.\n'
            'This matrix is very close to example 5.\n'
            'Mathematically, it has a complete set of eigenvectors.\n'
            'Numerically, however, the calculation will be suspect.',
            numpy.array([
                [1 + matlab_eps, 1],
                [0, 1 - matlab_eps],
                ], dtype=float),
            numpy.array([
                [2.718309, 2.718282],
                [0, 2.718255],
                ], dtype=float),
            ),

        ExpmCase(
            8,
            'This matrix was an example in Wikipedia.',
            numpy.array([
                [21, 17, 6],
                [-5, -1, -6],
                [4, 4, 16],
                ], dtype=float),
            numpy.array([
                [13*exp16 - exp4, 13*exp16 - 5*exp4,  2*exp16 - 2*exp4],
                [-9*exp16 + exp4, -9*exp16 + 5*exp4, -2*exp16 + 2*exp4],
                [16*exp16,        16*exp16,           4*exp16         ],
                ], dtype=float) * 0.25,
            ),

        ExpmCase(
            9,
            'This matrix is due to the NAG Library.\n'
            'It is an example for function F01ECF.',
            numpy.array([
                [1, 2, 2, 2],
                [3, 1, 1, 2],
                [3, 2, 1, 2],
                [3, 3, 3, 1],
                ], dtype=float),
            numpy.array([
                [740.7038, 610.8500, 542.2743, 549.1753],
                [731.2510, 603.5524, 535.0884, 542.2743],
                [823.7630, 679.4257, 603.5524, 610.8500],
                [998.4355, 823.7630, 731.2510, 740.7038],
                ], dtype=float),
            ),

        ExpmCase(
            10,
            'This is Ward\'s example #1.\n'
            'It is defective and nonderogatory.\n'
            'The eigenvalues are 3, 3 and 6.',
            numpy.array([
                [4, 2, 0],
                [1, 4, 1],
                [1, 1, 4],
                ], dtype=float),
            numpy.array([
                [147.8666224463699, 183.7651386463682, 71.79703239999647],
                [127.7810855231823, 183.7651386463682, 91.88256932318415],
                [127.7810855231824, 163.6796017231806, 111.9681062463718],
                ], dtype=float),
            ),

        ExpmCase(
            11,
            'This is Ward\'s example #2.\n'
            'It is a symmetric matrix.\n'
            'The eigenvalues are 20, 30, 40.',
            numpy.array([
                [29.87942128909879, 0.7815750847907159, -2.289519314033932],
                [0.7815750847907159, 25.72656945571064, 8.680737820540137],
                [-2.289519314033932, 8.680737820540137, 34.39400925519054],
                ], dtype=float),
            numpy.array([
                 [
                     5.496313853692378E+15,
                     -1.823188097200898E+16,
                     -3.047577080858001E+16],
                 [
                    -1.823188097200899E+16,
                    6.060522870222108E+16,
                    1.012918429302482E+17],
                 [
                    -3.047577080858001E+16,
                    1.012918429302482E+17,
                    1.692944112408493E+17],
                ], dtype=float),
            ),

        ExpmCase(
            12,
            'This is Ward\'s example #3.\n'
            'Ward\'s algorithm has difficulty estimating the accuracy\n'
            'of its results.  The eigenvalues are -1, -2, -20.',
            numpy.array([
                [-131, 19, 18],
                [-390, 56, 54],
                [-387, 57, 52],
                ], dtype=float),
            numpy.array([
                [-1.509644158793135, 0.3678794391096522, 0.1353352811751005],
                [-5.632570799891469, 1.471517758499875, 0.4060058435250609],
                [-4.934938326088363, 1.103638317328798, 0.5413411267617766],
                ], dtype=float),
            ),
        ]

class WrappedPade:
    def __init__(self, q):
        self.q = q
    def __call__(self, M):
        return algopy.expm_pade(M, self.q)


def main():
    f_name_pairs = (
            (WrappedPade(3), 'Pade q=3'),
            (WrappedPade(5), 'Pade q=5'),
            (WrappedPade(7), 'Pade q=7'),
            (WrappedPade(9), 'Pade q=9'),
            (WrappedPade(13), 'Pade q=13'),
            (algopy.expm_higham_2005, 'Higham 2005'),
            )
    for f, name in f_name_pairs:
        print(name)
        for case in g_burkardt_expm_cases:
            observed = f(case.M)
            expected = case.expm_M
            err_norm = numpy.linalg.norm(observed - expected, 1)
            rel_err = err_norm / numpy.linalg.norm(expected, 1)
            if rel_err < 1e-12:
                s_stars = '   '
            elif rel_err < 1e-8:
                s_stars = '*  '
            elif rel_err < 1e-4:
                s_stars = '** '
            elif rel_err < 1e4:
                s_stars = '***'
            else:
                s_stars = 'OMG'
            s_index = str(case.burkardt_index).rjust(2)
            print(s_index, ':', s_stars, rel_err)
        print()

if __name__ == '__main__':
    main()


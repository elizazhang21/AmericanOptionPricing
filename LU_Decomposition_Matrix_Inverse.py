# LU_Decomposition_Matrix_Inverse.py


import numpy as np
import scipy
from scipy.linalg import lu, lu_solve, lu_factor
from numpy.linalg import inv


def LU_Decomposition_Matrix_Inverse(A):
    A = scipy.array(A)
    b = np.ones(len(A))
    P, L, U = lu(A)  # A = PLU, P is permutation matrix
    L = np.dot(P, L)  # A = LU, L is lower triangle
    # print(L)
    # x = U^{-1}(L^{-1}*b)
    x = np.dot(inv(U), np.dot(inv(L), b))
    return x

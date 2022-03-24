import math
import numpy as np
import cupy as cp

# table 10.4
b_d = {
    3:  [120,60,12,1],
    5:  [30240,15120,3360,420,30,1],
    7:  [17297280, 8648640,1995840, 277200, 25200,1512, 56,1],
    9:  [17643225600,8821612800,2075673600,302702400,30270240, 2162160,110880,3960,90,1],
    13: [64764752532480000,32382376266240000,7771770303897600,1187353796428800,
                129060195264000, 10559470521600,670442572800,33522128640,1323241920,40840800,960960,16380,182,1]
}

# table 10.2
theta_dict = {3:0.015, 5:0.25, 7:0.95, 9:2.1, 13:5.4}


def _expm_pade(A:cp.ndarray, M:int):
    N0 = A.shape[0]
    dtype = A.dtype
    b = b_d[M]
    U = b[1]*cp.eye(N0, dtype=dtype)
    V = b[0]*cp.eye(N0, dtype=dtype)
    A2 = cp.dot(A,A)
    A2n = cp.eye(N0, dtype=dtype)
    for i in range(1,M//2+1):
        A2n = cp.dot(A2n, A2)
        U += b[2*i+1] * A2n
        V += b[2*i] * A2n
    U = cp.dot(A,U)
    ret = cp.linalg.solve(V-U, V+U)
    return ret


def _expm_ss(A, norm):
    # algorithm 10.20, from line 7
    N0 = A.shape[0]
    b = b_d[13]
    s = max(0, int(math.ceil(math.log(norm/theta_dict[13])/math.log(2))))
    if s > 0:
        A = A/2**s #NOT modify the argument A
    Id = cp.eye(N0)
    A2 = cp.dot(A, A)
    A4 = cp.dot(A2, A2)
    A6 = cp.dot(A2, A4)
    U = cp.dot(A, cp.dot(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*Id)
    V = cp.dot(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*Id
    r13 = cp.linalg.solve(V-U, V+U)
    ret = cp.linalg.matrix_power(r13, 2**s)
    return ret


def expm(A):
    """
    Calculate the matrix exponential of a square matrix A: MatrixExp(A)

    This module implements algorithm 10.20 from [1]. The matrix exponential is
    calculated using scaling and squaring, and a Pade approximation.

    [1] Functions of Matrices: Theory and Computation, Nicholas J. Higham, 2008
    """
    rows, columns = A.shape
    assert (A.ndim==2) and (A.shape[0]==A.shape[1])

    norm = cp.linalg.norm(A, ord=1)

    if norm < theta_dict[3]:
        ret = _expm_pade(A, 3)
    elif norm < theta_dict[5]:
        ret = _expm_pade(A, 5)
    elif norm < theta_dict[7]:
        ret = _expm_pade(A, 7)
    elif norm < theta_dict[9]:
        ret = _expm_pade(A, 9)
    else:
        ret = _expm_ss(A, norm)
    return ret

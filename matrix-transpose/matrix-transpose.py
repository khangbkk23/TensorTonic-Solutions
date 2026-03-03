import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.asarray(A, dtype=float)
    m, n = A.shape

    result = np.zeros((n,m), dtype=float)

    for i in range(m):
        for j in range(n):
            result[j][i] = A[i][j]

    return result

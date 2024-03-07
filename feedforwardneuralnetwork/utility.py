import numpy as np

def matrix_vector_multiply(M, V):
    return np.dot(M,V)

def hadamard_product(A,B):
    result = []
    for i in range(len(A)):
        result.append(A[i]*B[i])

    return result
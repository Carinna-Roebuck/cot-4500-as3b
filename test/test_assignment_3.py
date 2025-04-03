import numpy as np
from scipy.linalg import lu

def gaussian_elimination(A, b):
    n = len(b)
    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))
    
    # Forward elimination
    for i in range(n):
        pivot = augmented_matrix[i, i]
        augmented_matrix[i] = augmented_matrix[i] / pivot
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i]
            augmented_matrix[j] -= factor * augmented_matrix[i]
    
    # Backward substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = augmented_matrix[i, -1] - np.sum(augmented_matrix[i, i+1:n] * x[i+1:n])
    
    return x

def lu_factorization(A):
    P, L, U = lu(A)
    return L, U

def is_diagonally_dominant(A):
    n = A.shape[0]
    for i in range(n):
        diagonal_element = abs(A[i, i])
        off_diagonal_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diagonal_element < off_diagonal_sum:
            return False
    return True

def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

# Test matrices
A1 = np.array([[3, -2, 5],
               [1, 4, -1],
               [2, -3, 3]], dtype=float)
b1 = np.array([7, -3, 5], dtype=float)

A2 = np.array([[4, 2, 1, 3],
               [3, 3, -2, 1],
               [2, 1, -1, 4],
               [1, 2, 3, 5]], dtype=float)

A3 = np.array([[10, 2, 1, 1, 1],
               [3, 12, 2, 3, 2],
               [1, 2, 8, 2, 1],
               [2, 3, 2, 15, 4],
               [3, 2, 4, 1, 10]], dtype=float)

A4 = np.array([[3, -1, 1],
               [-1, 5, 2],
               [1, 2, 4]], dtype=float)

# Solve systems and print results
solution1 = gaussian_elimination(A1, b1)
print("Solution to Gaussian Elimination:", solution1)

L, U = lu_factorization(A2)
det2 = np.linalg.det(A2)
print("Determinant:", det2)
print("L matrix:\n", L)
print("U matrix:\n", U)

solution3 = is_diagonally_dominant(A3)
print("Is the matrix diagonally dominant?:", solution3)

solution4 = is_positive_definite(A4)
print("Is the matrix positive definite?:", solution4)

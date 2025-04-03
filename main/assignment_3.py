#Question 1
import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))
    
    # Forward elimination
    for i in range(n):
        # Make the diagonal element 1
        pivot = augmented_matrix[i, i]
        augmented_matrix[i] = augmented_matrix[i] / pivot
        
        # Make the elements below the pivot 0
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i]
            augmented_matrix[j] -= factor * augmented_matrix[i]
    
    # Backward substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = augmented_matrix[i, -1] - np.sum(augmented_matrix[i, i+1:n] * x[i+1:n])
    
    return x

# Define the augmented matrix
A = np.array([[2, -1, 1],
              [1, 3, 1],
              [-1, 5, 4]], dtype=float)
b = np.array([6, 0, -3], dtype=float)

# Solve the system
solution = gaussian_elimination(A, b)
print(solution)

#Question 2
from scipy.linalg import lu

def lu_factorization(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float)
    
    for i in range(n):
        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j] -= factor * U[i]
    
    return L, U

# Define the matrix
A = np.array([[1, 1, 0, 3],
              [2, 1, -1, 1],
              [3, -1, -1, 2],
              [-1, 2, 3, -1]], dtype=float)

# Compute LU factorization
L, U = lu_factorization(A)

det = np.linalg.det(A)

# Print results
print("Determinant:", det)
print("L matrix:\n", L)
print("U matrix:\n", U)


# Question 3
def is_diagonally_dominant(A):
    n = A.shape[0]
    for i in range(n):
        diagonal_element = abs(A[i, i])
        off_diagonal_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diagonal_element < off_diagonal_sum:
            return False
    return True

# Define the matrix
A = np.array([[9, 0, 5, 2, 1],
              [3, 9, 1, 2, 1],
              [0, 1, 7, 2, 3],
              [4, 2, 3, 12, 2],
              [3, 2, 4, 0, 8]], dtype=float)

# Check if the matrix is diagonally dominant
solution = is_diagonally_dominant(A)
print("Is the matrix diagonally dominate?:",solution)

# Question 4

def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

# Define the matrix
A = np.array([[2, 2, 1],
              [2, 3, 0],
              [1, 0, 2]], dtype=float)

# Check if the matrix is positive definite
pos_def = is_positive_definite(A)

# Print result
print("Is the matrix positive definite?:", pos_def)
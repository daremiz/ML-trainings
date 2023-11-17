#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
def power_method(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    n = data.shape[0]

    # Initialize a random vector (you can also use np.ones for simplicity)
    b_k = np.random.rand(n)

    for _ in range(num_steps):
        # Apply the matrix to the vector
        Ab_k = np.dot(data, b_k)
        
        # Update the vector
        b_k = Ab_k / np.linalg.norm(Ab_k)

    # Estimate the dominant eigenvalue
    eigenvalue = np.dot(b_k, np.dot(data, b_k))

    return eigenvalue, b_k


def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    eigenvalue, eigenvector = power_method(data, num_steps)
    return float(eigenvalue), eigenvector

    pass


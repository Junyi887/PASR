import numpy as np
import torch
import h5py

def generate_toeplitz(cols:int, final_index:int):
# Calculate the number of rows based on the final index and number of columns
    rows = final_index - cols + 2
    
    # Initialize a matrix filled with zeros
    matrix = np.zeros((rows, cols))
    
    # Fill the matrix such that it becomes a Toeplitz matrix
    for i in range(rows):
        for j in range(cols):
            value = i + j
            matrix[i, j] = min(value, final_index)
                
    return matrix

def generate_test_matrix(cols:int, final_index:int):
    # Calculate the number of rows based on the final index and number of columns
    rows = (final_index + 1) // (cols - 1)
    
    # Check if an additional row is needed to reach the final index
    if (final_index + 1) % (cols - 1) != 0:
        rows += 1
    
    # Initialize a matrix filled with zeros
    matrix = np.zeros((rows, cols))
    
    # Fill the matrix according to the specified pattern
    current_value = 0
    for i in range(rows):
        for j in range(cols):
            if current_value <= final_index:
                matrix[i, j] = current_value
                current_value += 1
        current_value -= 1  # Repeat the last element in the next row
                
    return matrix[:-1,:]
import numpy as np
import sys
import time
from scipy.linalg import expm

# Defining constants
eps = sys.float_info.epsilon  # Machine epsilon for floating point precision
max_fact = 170  # Maximum n where factorial(n) < float max value
n = 10  # sample size
error_margin = 1e-5  # acceptable error margin

# Function to load a matrix from a file
def matrix_from_file(filename):
    """
    Load a matrix from a file.
    """
    matrix = []
    with open(filename, 'r') as f:
        for line in f:
            row = [float(num) for num in line.split(',')]
            matrix.append(row)
    return np.array(matrix)

# Function to check if a matrix is square
def validate_square(matrix):
    """
    Check if a matrix is square.
    """
    num_rows, num_cols = matrix.shape
    return num_rows == num_cols

# Function to write the resultant matrix to a file
def result_to_file(result_matrix, filename):
    """
    Write the resultant matrix to a file.
    """
    with open(filename, 'w') as f:
        for row in result_matrix:
            line = ",".join(str(num) for num in row)
            f.write(line + '\n')

def lagrange(x, y, xi):
    """
    Lagrange Polynomial Interpolation.
    """
    n = len(x)
    yi = 0
    for i in range(n):
        p = 1
        for j in range(n):
            if i != j:
                p *= (xi - x[j]) / (x[i] - x[j])
        yi += y[i] * p
    return yi

# Function to calculate the exponential of a matrix
def calculate_exponential(matrix):
    """
    Calculate the exponential of a matrix.
    """
    iter_num, result = 0, np.zeros(matrix.shape)
    x, y = [], []  # sample points for x and y

    matrix_power = np.eye(matrix.shape[0])  # Initialize matrix_power
    fact = 1  # Initialize factorial

    while True:
        if iter_num > 0:
            matrix_power = np.dot(matrix_power, matrix)  # Update matrix_power
        matrix_factor = matrix_power / fact  # Calculate matrix_factor
        result = result + matrix_factor  # Update result

        if len(x) == n:
            x.pop(0)  # Remove the oldest value from x
            y.pop(0)  # Remove the oldest value from y
        x.append(iter_num)  # Add the current iteration number to x
        y.append(matrix_factor.sum())  # Sum the elements of matrix_factor and add to y

        if len(x) == n:
            estimate = lagrange(x, y, iter_num + 1)
            if np.abs(estimate - y[-1]) < error_margin:
                break

        iter_num += 1
        fact *= iter_num

        if iter_num > max_fact:
            fact = 1e100

    return result

# Main function
def main():
    """
    Main function of the program.
    """
    print("Matrix Exponential Calculation Program\n" + "-"*50)
    # Load the input matrix from a file
    input_matrix = matrix_from_file('inv_matrix(1000x1000).txt')
    # Check if the matrix is a square matrix
    if not validate_square(input_matrix):
        print("Invalid input. The input matrix must be square.")
        return
    # Compute the exponential of the matrix
    start_time = time.perf_counter()
    result_matrix = calculate_exponential(input_matrix)
    end_time = time.perf_counter()
    print(f"\nCustom implementation took: {end_time - start_time:.6f} seconds.")
    print(f"Top-left and bottom-right elements of the resultant matrix are: [{result_matrix[0][0]}, {result_matrix[0][1]}, ..., {result_matrix[-1][-2]}, {result_matrix[-1][-1]}]")
    # Save the result matrix to a file
    result_to_file(result_matrix, 'result.txt')
    # Compare with numpy's implementation
    start_time = time.perf_counter()
    result_matrix_numpy = expm(np.array(input_matrix))
    end_time = time.perf_counter()
    print(f"\nNumpy's implementation took: {end_time - start_time:.6f} seconds.")
    print(f"Top-left and bottom-right elements of the resultant matrix are: [{result_matrix_numpy[0][0]}, {result_matrix_numpy[0][1]}, ..., {result_matrix_numpy[-1][-2]}, {result_matrix_numpy[-1][-1]}]")
    print("\n" + "-"*50)
    print("Execution completed successfully. Check the 'result.txt' file for the output matrix from the custom implementation.")

# Entry point of the program
if __name__ == "__main__":
    main()
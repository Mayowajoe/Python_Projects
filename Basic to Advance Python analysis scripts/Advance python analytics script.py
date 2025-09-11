# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:19:04 2024

@author: adebo
"""

import numpy as np
from math import e
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.optimize import linprog
import seaborn as sns
sns.set(style="darkgrid")

## QUESTION 1

# Define the functions
def calculate_matrix_operations():
    # Create a matrix with 20 rows and 10 columns (elements are random integers from 1 to 20)
    X = np.random.randint(1, 21, (20, 10))

    # Create a column matrix of 20 integers from 1 to 20
    Y = np.arange(1, 21).reshape(-1, 1)

    # Find the transpose of matrix X
    XT = X.T

    # Calculate (X^T * X)^(-1) * X^T * Y
    XTX = XT @ X

    # Handling potential singularity by checking if the matrix is invertible
    if np.linalg.det(XTX) != 0:
        XTX_inv = np.linalg.inv(XTX)
        result = XTX_inv @ XT @ Y
        print("\nResult of (X^T * X)^(-1) * X^T * Y:\n", result)
    else:
        print("\nMatrix (X^T * X) is singular and cannot be inverted.")

# Call the function
calculate_matrix_operations()


## QUESTION 2
# Define the function
def f_w_x_y_z(w, x, y, z):
    return np.exp(x) + np.sin(np.radians(y)) - np.log10(w) - (z ** 4) / (x + y + w)

# Evaluate f(w, x, y, z) for w=300, x=10, y=60, z=5
w = 300
x = 10
y = 60
z = 5
f_w_x_y_z_result = f_w_x_y_z(w, x, y, z)
print("f(w, x, y, z) result:", f_w_x_y_z_result)

## QUESTION 3
def evaluate_series():
    # Initialize the result with the first term, which is 1
    result = 1

    # Set up the initial values for the product term
    term = 3 / 4
    result += term

    # Continue adding terms as per the given pattern
    for i in range(6, 34, 2):
        term *= (i - 1) / i
        result += term

    # Print the final result of the evaluation
    print(result)

# Call the function
evaluate_series()

## ANOTHER METHOD

result = 1

# Loop to generate and add the rest of the terms
current_product = 1
for i in range(3, 33, 2):
    current_product *= (i / (i + 1))
    result += current_product

print(result)

## QUESTION 4

# Double summation code for the given problem
result = 0

# Outer summation over x from 1 to 20
for x in range(1, 21):
    # Inner summation over y from 1 to x
    for y in range(1, x + 1):
        result += (x**4) / (3 + y**5)

print(result)


## QUESTION 5

# Step 1: Define the piecewise function
def tmpFn(x):
    # Use numpy's piecewise to define the function
    f = np.piecewise(x, 
                     [x < 0, (x >= 0) & (x < 5), x >= 5], 
                     [lambda x: x**2 + 5*x + 9, 
                      lambda x: x + 10, 
                      lambda x: x**2 + 4*x - 17])
    return f

# Step 2: Create values for x in the range -10 < x < 10
x = np.linspace(-10, 10, 400)

# Step 3: Evaluate the function at these values
y = tmpFn(x)

# Step 4: Plot the function


# Step 4: Plot the function using Seaborn
# sns.set(style="whitegrid")  # Set the style to white grid for better aesthetics

plt.figure(figsize=(10, 6))
sns.lineplot(x=x, y=y, label='f(x)', color='b')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of the Function f(x)')
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.grid(True)
plt.show()




plt.figure(figsize=(10, 6))
plt.plot(x, y, label='f(x)', color='b')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of the Function f(x)')
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.grid(True)
plt.show()


## QUESTION 6

# Part 1: Integration
def integrand(x, y):
    return 3 * np.log(x) + np.exp(-y)

# Perform the double integral
result, error = dblquad(integrand, 0, 100, lambda y: 2, lambda y: 500)

# Part 2: Matrix generation and computation
random.seed(42)  # Set the random seed for reproducibility
matrix = np.random.randint(1, 21, size=(10, 10))

# Output the results
result, matrix

## QUESTION 7

entries_greater_than_5 = np.sum(matrix > 5, axis=1)
entries_greater_than_5 

## QUESTION 8
rows_with_two_sixes = [i + 1  for i, row in enumerate(matrix) if np.count_nonzero(row == 6) == 2]
rows_with_two_sixes 


## QUESTION 9
rows_with_sum_greater_than_50 = np.sum(np.sum(matrix, axis=1) > 50)
rows_with_sum_greater_than_50

## QUESTION 10
row_sums = np.sum(matrix, axis=1)
row_sums

highest_sum_row = np.argmax(row_sums) + 1 # To ensures that the row indices will be 1-based instead of 0-based.
highest_sum_row

lowest_sum_row = np.argmin(row_sums) + 1 # To ensures that the row indices will be 1-based instead of 0-based.
lowest_sum_row

## QUESTION 11

# Coefficients of the objective function (to minimize -2x - 5y)
c = [2, 5]

# Coefficients of the inequality constraints (Ax <= b)
A = [[1, 4],  # x + 4y <= 24
     [3, 1],  # 3x + y <= 21
     [1, 1]]  # x + y <= 9

# RHS of the inequality constraints
b = [24, 21, 9]

# Bounds for x and y (x >= 0, y >= 0)
x_bounds = (0, None)
y_bounds = (0, None)

# Solve the linear programming problem
res = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds], method='simplex')

# Extract the results
res.fun, res.x  # Objective function value (maximized Z) and values of x and y

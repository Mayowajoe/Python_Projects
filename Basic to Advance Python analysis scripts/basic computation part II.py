#-----------------------------
# Basic Co mputation Part II
#-----------------------------
# Following packages are needed to conduct some computation
import numpy as np
from math import e
from math import inf
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

# Q1: Writing function for 1, 2 and 3 arguments
#a

x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])
z=np.array([1,2,3,4,5,6,7,8,9,10])

def f1a(x):
    result=5*x+10
    return result

f1a(x)

#b

def f1b(x,y):
    result=x+np.exp(x)-np.log(y)
    return result

f1b(x,y)

#c
def f1c(x,y,z):
    result=np.pi*(np.exp(x))-17*y+np.log(z)
    return result

f1c(x,y,z) 

# Q2
import sympy as sym
# a
solution_a = sym.solve('3*x + 5','x')
solution_a[0]
#b
solution_b = sym.solve('x**2 - 2 * x + 1','x')
solution_b[0]
#c
solution_c = sym.solve('x**2 + 4 * x +3 ','x')
solution_c[0]
#d
solution_d = sym.solve('4*x**2 - 3 * x +19 ','x')
solution_d[0]

#Q3
A = np.array([[7,5,4],
              [9,8,5],
              [2,7,4]])

B = np.array([[9,3,6],
              [7,5,5],
              [1,4,5]])
# rows of matrix D are not linearly independent
D = np.array([[9,3,6],
              [3,1,2],
              [1,4,5]])
# a
A + B

# b
A -B

# c
# elementwise multipication
A*B
# matrix multiplication
A.dot(B)

# d
np.linalg.det(A)
# e
np.linalg.inv(A)
# f
np.linalg.inv(B)
# g
np.linalg.matrix_rank(A)

np.linalg.matrix_rank(D)
# h
A.transpose()
# Q4
P = np.array([[2,3],
              [5,7],
              [4,9]])

Q = np.array([[5,6,8],
              [7,4,1]])
# a
P.dot(Q)

# b
P * 9

# Q5
I = np.identity(3)
I

# Q6
unit_matrix = np.array([1]*9).reshape(3,3)
unit_matrix

# Q7
zero_matrix = np.zeros(9).reshape(3,3)
zero_matrix

# Q8
from sympy import *
# a
x, y = sym.symbols(('x','y'))

eq_system=sym.Matrix([[3,4,5],
                      [7,-3,7]])

solution = linsolve(eq_system,(x,y))
x,y = next(iter(solution))
solution

# b
x, y, z = sym.symbols(('x','y','z'))
eq_system=sym.Matrix([[7,9,6,87],
                      [8,3,7,77],
                      [9,8,-7,13]])
solution = linsolve(eq_system,(x,y,z))
x,y,z = next(iter(solution))
solution

# Q9
i = np.linspace(1,100,100)
sum(i)

# Q10
sum(i**2)

# Q11
sum(i-np.mean(i))

# Q12
sum((i-np.mean(i))**2)

# Q13
sum(i ** 3 + 4 * i ** 2)
# Q14
sum((2 ** i) / i + (3 ** i) / (i **2))
# Q15
sum(e ** (-i + 1) / (i + 10))
# Q16
random.seed(110) 
# random.seed generate same numbers each time the code is run:
x = np.array([random.randint(0,999) for i in range(100)])

# this solution assumes X_i+1 is the current value of x + 1 and not the next random integer int the sequence

sum(e ** (-x + 1) / (x + 10)) 

# Q17

def f17(n):
    a_i=1

    # the loop doesn't occur if n=0, and a_1 remains equal to 1 as a result
    for i in range(n):

        # record the previous value of a_i before the current index
        a_i_minus_1=a_i 

        # position in series before summation
        x_num=2*(i+1) # translate the zero indexed sequence position into numerical positions (1=1st element in series) and then multiply by 2 to get numerator
        x_denom=x_num+1

        # updating the value of a at index i
        a_i=a_i_minus_1*(x_num/x_denom) 

    return(a_i)   

# python is 0 indexed: 39 is the number to stop at and isn't an input to
# the sequence

# process 20 numbers in a sequence: actually, numbers 0 to 19, because python is zero indexed
#instead of range(20), we could have also used np.linspace(1,19).astype(int)
series = [f17(i) for i in range(20)] 
sum(series)


# Q18
i_series = np.linspace(1,20,20).astype(int) # or i_series=range(1,21) # zero indexed programming language
j_series = np.linspace(1,5,5).astype(int) # or j_series=range(1,6)
series=[i ** 4 / (3 + j) for j in j_series for i in i_series] # order of loops is unimportant
sum(series)

# alternative
total = 0
# ranges adjusted to reflect actual values of i and j
for i in range(1,21):
    for j in range(1,6):
        total += (i ** 4) / (3 + j)
total
    
# Q19

total = 0
for i in range(1,11):
    for j in range(1,i+1):
        total += (i ** 4) / (3 + (i*j))
total
# alternative:
total=sum([sum([i**4/(3+i*j) for j in range(1,i+1)]) for i in range(1,11)])



# Q20
x = np.arange(0,1001)
total = sum(e ** -x)

# Q21
import numbers
def foo1(x,n):
    # gives the error message to the right if 1) not a positive number
    # or 2) not an integer
    assert isinstance(x, numbers.Real), "x must be a real number"
    assert n > 0, "This function does not accept a number n <= 0"
    assert isinstance(n,int), "This function does not accept non-integer values"
    if n == 1:
        return 1
    result = [1] + [x**(i)/(i) for i in range(1,n+1)]
    return result
total = foo1(10,10)
sum(total)

# Q22
x = np.linspace(-2.99,2.99,1000) # make 1000 evenly spaced values from -2.99 to 2.99
def tmpFn(xi):
        if xi < 0:
            return xi ** 2 + 2 * xi + 3
        
        elif xi >= 0 and xi < 2:
            return xi + 3
        
        elif xi >= 2:
            return xi**2+4*xi-7
        
tmpFn = np.vectorize(tmpFn) # when you input a list or array, the new function will apply the function to each element within it
                            # similar to how a for loop works
fx = tmpFn(x)

mpl.style.use("seaborn")
plt.plot(x, fx)

# Q23: same as Q19

def foo1(n):  
    result=sum([(s ** 2) / (10 + 4*(r**3)) for r in range(1,n+1) for s in range(1,r+1)])
    return (result)

# alternative:
def foo1(n):
    result=sum([
    sum([(s ** 2) / (10 + 4*(r**3)) for s in range(1,r+1)]) for r in range(1,n+1)
])
    return result

foo1(10)
      

# Q24

n=100
x=np.array([random.randint(0,999) for i in range(n)])
y=np.array([random.randint(0,999) for i in range(n)])

#a
y[1:] - x[:-1] # take elements 2 through n from y, 1 to n - 1 from x, and subtract by index

#b
np.sin(y[:-1]) / np.cos(x[1:])

#c
# alternating positive and negative numbers: positive if index i is even, else negative 

def get_coef(i):
    if i % 3 == 0:
        return 1
    elif i %  3 == 1:
        return 2
    else:
        return -1
sign=[get_coef(i) for  i in range(n)]
sign*x
sum(sign*x)


# d
sum(np.exp(-x[1:]) / (x[:-1] + 10))
# Q25

# a
A=np.random.randint(low=1,high=11,size=60).reshape(6,10) # 0 - 10, (not inclusive of 11 )

total_greater_4 = (A>4).sum(axis=1) # A > 4 gives 1 if a_i > 4 and 0 otherwise. Sum of each row = the total amount of random integers greater than 4
total_greater_4

# b
total_7s_by_row=(A==7).sum(axis=1) # A == 7 gives 1 if a_i =7 and 0 otherwise. Sum of each row = the total amount of random integers greater than 4
sum(total_7s_by_row == 2) # total number of rows with 2 7s

# Q26
import scipy.integrate as spi

integrand = lambda x : x**2  + np.exp(x)
a = 2
b = 10
result, error = spi.quad(integrand, a, b)
result
error

# Q27
integrand = lambda x : x**4 + 5*np.exp(-5*x)
a = 0
b = inf
result, error = spi.quad(integrand, a, b)
result
error

# Q28
integrand = lambda x : 3*np.log(x)
a = 100
b = 1000
result, error = spi.quad(integrand, a, b)
result
error
# Q29
integrand = lambda x : (x**2 + np.exp(x)+np.log(x))/((np.pi)*(np.exp(x)))
a = 50
b = 100
result, error = spi.quad(integrand, a, b)
result
error
# Q30
integrand = lambda x : np.sin(x)*(1+np.exp(x)+np.log(x))
a = 5
b = 20
result, error = spi.quad(integrand, a, b)
result
error
# Q31
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# drawing figures
fig, ax = plt.subplots(figsize=(8, 6))
ax.grid()

# Draw constraint lines
ax.hlines(0, -1, 17.5)
ax.vlines(0, -1, 12)
ax.plot(np.linspace(-1, 17.5, 100), 6-0.4*np.linspace(-1, 17.5, 100), color="c")
ax.plot(np.linspace(-1, 5.5, 100), 10-2*np.linspace(-1, 5.5, 100), color="c")
ax.text(1.5, 8, "$2x_1 + 5x_2 \leq 30$", size=12)
ax.text(10, 2.5, "$4x_1 + 2x_2 \leq 20$", size=12)
ax.text(-2, 2, "$x_2 \geq 0$", size=12)
ax.text(2.5, -0.7, "$x_1 \geq 0$", size=12)

# Draw the feasible region
feasible_set = Polygon(np.array([[0, 0],
                                 [0, 6],
                                 [2.5, 5],
                                 [5, 0]]),
                       color="cyan")
ax.add_patch(feasible_set)

# Draw the objective function
ax.plot(np.linspace(-1, 5.5, 100), 3.875-0.75*np.linspace(-1, 5.5, 100), color="orange")
ax.plot(np.linspace(-1, 5.5, 100), 5.375-0.75*np.linspace(-1, 5.5, 100), color="orange")
ax.plot(np.linspace(-1, 5.5, 100), 6.875-0.75*np.linspace(-1, 5.5, 100), color="orange")
ax.arrow(-1.6, 5, 0, 2, width = 0.05, head_width=0.2, head_length=0.5, color="orange")
ax.text(5.7, 1, "$z = 3x_1 + 4x_2$", size=12)

# Draw the optimal solution
ax.plot(2.5, 5, "*", color="black")
ax.text(2.7, 5.2, "Optimal Solution", size=12)
plt.show()




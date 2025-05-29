#Given a vector x of length m, and a vector y of length n, compute m × n matrices: A and B, such
#that A(i, j) = x(i) + y(j), and B(i, j) = x(i) · y(j).

import numpy as np
m=2
n=3
x=np.random.rand(m)
y=np.random.rand(n)

A=x[:, np.newaxis] * np.ones(n)
A=A+y.T
B=np.outer(x,y)
print(A)
print(B)
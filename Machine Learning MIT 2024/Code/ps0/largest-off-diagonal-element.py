#Given an n × n matrix A, find the value of the largest off-diagonal element.
import numpy as np

n=4
A= np.random.rand(n,n)
print(A)
off_diagonal=A[~np.eye(A.shape[0],dtype=bool)]
print(off_diagonal)
max=np.max(off_diagonal)
print(max)
import numpy as np

#Given an n × n matrix C, add a scalar a to each diagonal entry of C.

n=5
C= np.random.rand(n,n)
s=np.eye(C.shape[0])

added_M=C+s
print(C)
print(added_M)

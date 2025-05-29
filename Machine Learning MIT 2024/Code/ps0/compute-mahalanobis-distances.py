'''The Mahalanobis distance is a measure of the distance between a point P and a distribution D,
introduced by P. C. Mahalanobis in 1936. It is a multi-dimensional generalization of the idea of
measuring how many standard deviations away P is from the mean of D. This distance is zero
if P is at the mean of D, and grows as P moves away from the mean: Along each principal
component axis, it measures the number of standard deviations from P to the mean of D. If
each of these axes is rescaled to have unit variance, then Mahalanobis distance corresponds to
standard Euclidean distance in the transformed space. Mahalanobis distance is thus unitless and
scale-invariant, and takes into account the correlations of the data set (from http://en.wikipedia.
org/wiki/Mahalanobis distance). Given a center vector c, a positive-definite covariance matrix S,
and a set of n vectors as columns in matrix X, compute the distances of each column in X to c,
using the following formula:
D(i) = (xi-c).T S^-1 (xi-c)  
Here, D is a vector of length n.'''
import numpy as np
n = 4
c = np.random.randn(n)
# we need a PD covariance matrix S,
# here's one (common) way of creating a random PD matrix
_ = np.random.randn(n, n)
S = (_) @ (_.T) + 1e-3 * np.eye(n)
X = np.random.randn(n, n)

# Solution 1: naive solution as baseline
# Not good: inversing S for n times.
D = np.zeros(n)
for i in range(n):
    z = X[:, i] - c
    D[i] = np.dot(np.dot(z.T, np.linalg.inv(S)), z)
# Solution 2: do pre-computation
D = np.zeros(n)
invS = np.linalg.inv(S)
for i in range(n):
    z = X[:, i] - c
    D[i] = np.dot(np.dot(z.T, invS), z)
# Solution 3: vectorization
Z = X - c[:, np.newaxis]
invS = np.linalg.inv(S)
D = np.sum(Z.conj() * (np.dot(invS, Z)), axis=0)
# Solution 4: directly solving linear equations is
# more efcient than doing inverse.
Z = X - c[:, np.newaxis]
D = np.sum(Z.conj() * (np.linalg.solve(S, Z)), axis=0)   #something like : S^-1 Z = H >> S H =Z >> np.linalg.solve(S,Z)
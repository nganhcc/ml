import numpy as np
import matplotlib.pyplot as plt

N=1000
mean=[4,2]
sigma=[[1,1.5],[1.5,3]]   #covariance

R= np.random.multivariate_normal(mean,sigma, N)
plt.plot(*(zip(*R)),marker=".", ls="")
plt.show()

muhat=np.mean(R,axis=0)
R_0=R- muhat[np.newaxis,:]
# Why is the denominator (N-1) instead of (N)? Check
# out Bessel's correction
sigma_hat=R_0.T.dot(R_0)/(N-1)
#find eigns
L,Q=np.linalg.eig(sigma_hat)
plt.arrow(
    muhat[0],
    muhat[1],
    Q[0, 0],
    Q[1, 0],
    shape="full",
    lw=3,
    length_includes_head=True,
    head_width=0.01,
)
plt.arrow(
    muhat[0],
    muhat[1],
    Q[0, 1],
    Q[1, 1],
    shape="full",
    lw=3,
    length_includes_head=True,
    head_width=0.01,
)
plt.plot(*(zip(*R)), marker=".", ls="")
plt.axis([1, 6, -2, 6])
plt.show()
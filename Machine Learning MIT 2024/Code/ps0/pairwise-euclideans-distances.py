import numpy as np

#find |xi| |yj| |xiyj|
d=4
m=3
n=5
x= np.random.rand(d,m)
y= np.random.rand(d,n)
xi_s=np.sum(x**2, axis=0)
yj_s=np.sum(y**2, axis=0)
xi_yj_s=x.T.dot(y)*2
broscast_xi_s=xi_s[:, np.newaxis]
eu_distances=np.sqrt(broscast_xi_s+yj_s.T-xi_yj_s)


tx2 = np.sum(x**2, 0)
ty2 = np.sum(y**2, 0)
Txy = np.dot(x.T, y)
D = tx2[:, np.newaxis] + ty2[np.newaxis, :] - 2 * Txy
D = np.sqrt(D)

print(eu_distances)
print(D)
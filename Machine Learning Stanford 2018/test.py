import numpy as np

# Example data: 4 points, 3D
X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
])
mu = np.array([3.0, 4.0, 5.0])  # Mean vector in 3D
cov = np.array([  # 3x3 covariance matrix
    [1.0, 0.2, 0.1],
    [0.2, 1.5, 0.3],
    [0.1, 0.3, 2.0]
])
cov_inv = np.linalg.inv(cov)  # Inverse of the covariance matrix

# Compute differences
diff = X - mu  # Shape (4, 3), 4 points in 3D space

# Quadratic form for each point
quadratic_form = np.sum(diff @ cov_inv * diff, axis=1)

print("Quadratic Form:", quadratic_form)
